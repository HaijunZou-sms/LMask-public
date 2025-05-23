# LMask
<details>
    <summary><strong>Overview</strong></summary>
<p align="center"><img src="./assets/LMask.png" width=95%></p>
</details>

## Installation
We  recommend installing the environment from the file by running the following commands
```bash
conda create -n lmask python=3.10
conda activate lmask
pip install -r requirements.txt
```
- Main Python packages:
  * PyTorch = 2.5.1
  * RL4CO = 0.5.2
  * TensorDict = 0.6.2
  * pytorch-lightning = 2.5.0 
  * PyVRP = 0.11.0
  * Numpy = 2.2.6
  * Pandas
  * tqdm
  

## Quickstart
### Generate Datasets
The validation and test datasets can be generated by running the following command:
```bash
python generate_datasets.py
```
### Test
* Test a specific random dataset
```bash
python driver/test.py --problem tsptw --problem_size 50 --hardness hard
```
If not provided, `test_path`, `checkpoint` and `ref_sol_path` will be automatically inferred from `problem`, `problem_size` and `hardness`. 
You can also provide additional parameters
```bash
usage: test.py [-h] [--seed SEED] [--batch_size BATCH_SIZE] [--problem {tspdl,tsptw}] [--problem_size {50,100}] [--hardness {easy,medium,hard}] 
               [--env_name ENV_NAME] [--policy_name POLICY_NAME] [--checkpoint CHECKPOINT] [--test_path TEST_PATH] [--ref_sol_path REF_SOL_PATH]
               [--look_ahead_step {1,2}] [--max_backtrack_steps MAX_BACKTRACK_STEPS] 
options:
  -h, --help            show this help message and exit
  --seed SEED
  --batch_size BATCH_SIZE
  --problem {tspdl,tsptw}
                        Problem type
  --problem_size {50,100}
                        Problem size
  --hardness {easy,medium,hard}
                        Problem difficulty
  --env_name ENV_NAME   Environment name
  --policy_name POLICY_NAME
                        Class name of the policy
  --checkpoint CHECKPOINT
                        Path to model checkpoint
  --test_path TEST_PATH
                        Path to test dataset
  --ref_sol_path REF_SOL_PATH
                        Path to reference solutions
  --look_ahead_step {1,2}
                        Number of lookahead steps when initializing the overestimation sets
  --max_backtrack_steps MAX_BACKTRACK_STEPS

```
* Test all datasets
```bash
python driver/test_all.py
```
The results will be saved to `results/main_results.csv`.

* Ablation study on the combination of backtracking and overestimation initialization strategies
```bash
python driver/bt_mask_comb.py
```
The results will be saved to `results/bt_mask_ablation.csv`.

### Training
```bash
python run.py experiment=main/tsptw/tsptw50-medium
```
You may change the experiment `experiment=main/tsptw/tsptw50-medium` by using the `experiment=YOUR_EXP`, with the path under [`configs/experiment`](configs/experiment) directory.

**Note**: After training, to use the checkpoints in test.py, you should first run the script `sripts/transform_checkpoints.py` to convert ckpt files to pth files.


### Datasets and Pre-trained Models
You can download datasets and pretrained models here: https://anonymous.4open.science/r/LMask-public-1C7F.


## Main Results

### TSPTW Results

| **Nodes** | **Method** | **n=50** |  |  |  |  | **n=100** |  |  |  |  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  |  | **Infeasible** |  | **Obj.** | **Gap** | **Time** | **Infeasible** |  | **Obj.** | **Gap** | **Time** |
|  |  | **Sol.** | **Ins.** |  |  |  | **Sol.** | **Ins.** |  |  |  |
| Easy | PyVRP | - | 0.00% | 7.31 | * | 1.7h | - | 0.00% | 10.19 | * | 4.3h |
|  | LKH3 | - | 0.00% | 7.31 | 0.00% | 1.9h | - | 0.00% | 10.21 | 0.29% | 7.2h |
|  | OR-Tools | - | 0.00% | 7.32 | 0.21% | 1.7h | - | 0.00% | 10.33 | 1.43% | 4.3h |
|  | Greedy-L | 31.84% | 31.84% | 9.49 | 30.28% | ≪1s | 33.23% | 33.23% | 13.92 | 36.80% | ≪1s |
|  | Greedy-C | 0.00% | 0.00% | 26.09 | 257.63% | ≪1s | 0.00% | 0.00% | 52.11 | 411.95% | ≪1s |
|  | PIP | 0.28% | 0.01% | 7.51 | 2.70% | 9s | 0.16% | 0.00% | 10.57 | 3.57% | 29s |
|  | PIP-D | 0.28% | 0.00% | 7.50 | 2.57% | 10s | 0.05% | 0.00% | 10.66 | 4.41% | 31s |
|  | LMask | **0.06%** | **0.00%** | **7.45** | **2.02%** | 7s | **0.01%** | **0.00%** | **10.50** | **3.11%** | 17s |
| Medium | PyVRP | - | 0.00% | 13.03 | * | 1.7h | - | 0.00% | 18.72 | * | 4.3h |
|  | LKH3 | - | 0.00% | 13.02 | 0.00% | 2.9h | - | 0.01% | 18.74 | 0.16% | 10.3h |
|  | OR-Tools | - | 15.12% | 13.01 | 0.12% | 1.5h | - | 0.52% | 18.98 | 1.40% | 4.3h |
|  | Greedy-L | 76.98% | 76.98% | 15.05 | 17.02% | ≪1s | 77.52% | 77.52% | 23.36 | 25.43% | ≪1s |
|  | Greedy-C | 42.26% | 42.26% | 25.40 | 96.45% | ≪1s | 18.20% | 18.20% | 51.69 | 176.58% | ≪1s |
|  | PIP | 4.82% | 1.07% | 13.41 | 3.07% | 10s | 4.35% | 0.39% | 19.62 | 4.73% | 29s |
|  | PIP-D | 4.14% | 0.90% | 13.46 | 3.45% | 9s | 3.46% | 0.03% | 19.80 | 5.70% | 31s |
|  | LMask | **0.06%** | **0.00%** | **13.25** | **1.73%** | 9s | **0.10%** | **0.00%** | **19.55** | **4.46%** | 20s |
| Hard | PyVRP | - | 0.00% | 25.61 | * | 1.7h | - | 0.01% | 51.27 | 0.00% | 4.3h |
|  | LKH3 | - | 0.52% | 25.61 | 0.00% | 2.3h | - | 0.95% | 51.27 | 0.00% | 1d8h |
|  | OR-Tools | - | 65.11% | 25.92 | 0.00% | 0.6h | - | 89.25% | 51.72 | 0.00% | 0.5h |
|  | Greedy-L | 70.94% | 70.94% | 26.03 | 0.29% | ≪1s | 93.17% | 93.17% | 52.20 | 0.29% | ≪1s |
|  | Greedy-C | 53.47% | 53.47% | 26.36 | 1.43% | ≪1s | 81.09% | 81.09% | 52.70 | 1.42% | ≪1s |
|  | PIP | 5.65% | 2.85% | 25.73 | 1.12% | 9s | 31.74% | 16.68% | 51.48 | 0.80% | 28s |
|  | PIP-D | 6.44% | 3.03% | 25.75 | 1.20% | 9s | 13.60% | 6.60% | 51.43 | 0.68% | 31s |
|  | LMask | **0.00%** | **0.00%** | **25.71** | **0.10%** | 6s | **0.00%** | **0.00%** | **51.38** | **0.21%** | 18s |


### TSPDL Results

| **Nodes** | **Method** | **n=50** |  |  |  |  | **n=100** |  |  |  |  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  |  | **Infeasible** |  | **Obj.** | **Gap** | **Time** | **Infeasible** |  | **Obj.** | **Gap** | **Time** |
|  |  | **Sol.** | **Ins.** |  |  |  | **Sol.** | **Ins.** |  |  |  |
| Medium | LKH | - | 0.00% | 10.85 | * | 2.3h | - | 0.00% | 16.36 | * | 10.2h |
|  | Greedy-L | 99.87% | 99.87% | 15.34 | 65.93% | ≪1s | 100.00% | 100.00% | - | - | ≪1s |
|  | Greedy-C | 0.00% | 0.00% | 26.12 | 144.33% | ≪1s | 0.00% | 0.00% | 52.14 | 222.79% | ≪1s |
|  | PIP | 1.75% | 0.17% | 11.23 | 5.09% | 8s | 2.50% | 0.16% | 17.68 | 9.39% | 21s |
|  | PIP-D | 2.29% | 0.22% | 11.27 | 5.44% | 8s | 1.83% | 0.23% | 17.80 | 10.12% | 23s |
|  | LMask | **0.03%** | **0.01%** | **11.14** | **2.75%** | 6s | **0.20%** | **0.05%** | **17.04** | **4.24%** | 15s |
| Hard | LKH | - | 0.00% | 13.25 | * | 2.6h | 0.00% | 0.00% | 20.76 | * | 15.8h |
|  | Greedy-L | 100.00% | 100.00% | - | - | ≪1s | 100.00% | 100.00% | - | - | ≪1s |
|  | Greedy-C | 0.00% | 0.00% | 26.09 | 100.25% | ≪1s | 0.00% | 0.00% | 52.16 | 155.38% | ≪1s |
|  | PIP | 4.83% | 2.39% | 13.63 | 4.49% | 8s | 29.34% | 21.65% | 22.35 | 9.71% | 20s |
|  | PIP-D | 4.16% | 0.82% | 13.79 | 5.68% | 8s | 13.51% | 8.43% | 22.90 | 12.57% | 23s |
|  | LMask | **0.19%** | **0.04%** | **13.57** | **2.52%** | 6s | **0.80%** | **0.26%** | **21.63** | **4.34%** | 15s |

### Ablation on the combination of backtracking and overestimation initialization strategies
<p float="left">
  <img src="assets/bt_mask_comb_gap.png" width="48%" />
  <img src="assets/bt_mask_comb_sol_infeas.png" width="48%" />
</p>