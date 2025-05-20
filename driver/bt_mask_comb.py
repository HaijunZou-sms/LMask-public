import os
import sys
import warnings

import pandas as pd
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

from driver.test import test_model_on_random_dataset
from lmask.utils.utils import infer_default_cofigs

# Define constants
PROBLEM = "tsptw"
PROBLEM_SIZE = 50
HARDNESS = "medium"
SEED = 2025
BATCH_SIZE = 2500
MAX_BACKTRACK_STEPS = 100


# Define model configurations
MODEL_CONFIGS = [
    {"model_name": "tsptw50-medium-bt-1step.pth", "trained_bt": True, "trained_step": 1},
    {"model_name": "tsptw50-medium-bt-2step.pth", "trained_bt": True, "trained_step": 2},
    {"model_name": "tsptw50-medium-nobt-1step.pth", "trained_bt": False, "trained_step": 1},
    {"model_name": "tsptw50-medium-nobt-2step.pth", "trained_bt": False, "trained_step": 2},
]

# Define test configurations
TEST_CONFIGS = [
    {"use_bt": True, "look_ahead": 1},
    {"use_bt": True, "look_ahead": 2},
    {"use_bt": False, "look_ahead": 1},
    {"use_bt": False, "look_ahead": 2},
]


def main():
    # Create the results directory if it doesn't exist
    result_dir = os.path.join(project_root, "results")
    Path(result_dir).mkdir(exist_ok=True, parents=True)
    result_file = os.path.join(result_dir, f"bt_mask_ablation.csv")

    # Get base configurations
    base_configs = infer_default_cofigs(PROBLEM, PROBLEM_SIZE, HARDNESS, SEED)

    # Prepare results list
    results = []

    # Run each combination
    for model_config in MODEL_CONFIGS:
        model_name = model_config["model_name"]
        model_path = os.path.join(project_root, "pretrained", "bt_ablation", model_name)

        for test_config in TEST_CONFIGS:
            print(f"\n{'='*80}")
            print(f"Testing model: {model_name}")
            print(f"Test config: use_bt={test_config['use_bt']}, look_ahead={test_config['look_ahead']}")
            print(f"{'='*80}")

            # Determine environment name based on whether backtracking is used
            env_name = f"{PROBLEM}-lazymask" if test_config["use_bt"] else PROBLEM

            # Run the test
            metrics = test_model_on_random_dataset(
                seed=SEED,
                env_name=env_name,
                policy_name=base_configs["policy_name"],
                checkpoint=model_path,
                batch_size=BATCH_SIZE,
                test_path=base_configs["test_path"],
                max_backtrack_steps=MAX_BACKTRACK_STEPS if test_config["use_bt"] else 0,
                look_ahead_step=test_config["look_ahead"],
            )

            # Add configuration information to the metrics
            sol_infeas_rate = 1 - metrics["sol_feas_rate"]
            ins_infeas_rate = 1 - metrics["ins_feas_rate"]
            avg_obj = - metrics["avg_reward"]  
            result_entry = {
                "model": model_name,
                "trained_bt": model_config["trained_bt"],
                "trained_step": model_config["trained_step"],
                "test_bt": test_config["use_bt"],
                "test_step": test_config["look_ahead"],
                "ins_infeas_rate":  f"{ins_infeas_rate: 2%}",
                "sol_infeas_rate": f"{sol_infeas_rate: 2%}",
                "avg_obj": f"{avg_obj: 2f}",
                "avg_gap": f"{metrics['avg_gap']: 2%}",
            }

            results.append(result_entry)

            # Save results incrementally in case of interruptions
            pd.DataFrame(results).to_csv(result_file, index=False)
            print(f"Updated results saved to {result_file}")

    print(f"\nAll tests completed. Final results saved to {result_file}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
