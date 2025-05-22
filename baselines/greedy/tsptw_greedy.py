from tqdm import tqdm
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)

import torch
from tensordict.tensordict import TensorDict
from rl4co.utils.ops import get_distance, gather_by_index

from lmask.utils.ops import get_tour_length

def get_tsptw_action_mask(td, look_ahead_step=2, round_error_epsilon=1e-5):
    if look_ahead_step == 1:
        curr_node = td["current_node"]

        # time window constraint
        d_ij = get_distance(gather_by_index(td["locs"], curr_node)[:, None, :], td["locs"])  # [B, n+1]
        arrival_time = td["current_time"][:, None] + d_ij
        can_reach_in_time = arrival_time <= (td["time_windows"][..., 1] + round_error_epsilon)  # [B, n+1]

        unvisited = ~td["visited"]

        can_visit_local = unvisited & can_reach_in_time  # [B, n+1]
        any_tw_viol = (unvisited & ~can_reach_in_time).any(dim=-1, keepdim=True)  # [B, 1]
        # If there is any node violating the time window, then we must backtrack
        can_visit = torch.where(any_tw_viol, torch.zeros_like(can_visit_local), can_visit_local)

    elif look_ahead_step == 2:
        batch_size, num_locs, _ = td["locs"].shape
        batch_idx = torch.arange(batch_size, device=td.device)  # [B, ]

        tw_early, tw_late = td["time_windows"].unbind(-1)

        dur_cur_succ = td["duration_matrix"][batch_idx, td["current_node"], :]

        service_start_time_succ = torch.max(td["current_time"].unsqueeze(1) + dur_cur_succ, tw_early)
        service_start_time_grandsucc = torch.max(service_start_time_succ.unsqueeze(-1) + td["duration_matrix"], tw_early.unsqueeze(1))  # Here dur_succ_grandsucc = distance_matrix

        succ_feasible = service_start_time_succ <= (tw_late + round_error_epsilon)  # [B, n+1]
        grandsucc_feasible = service_start_time_grandsucc <= (tw_late.unsqueeze(1) + round_error_epsilon)  # [B, n+1, n+1]

        eye = torch.eye(num_locs, dtype=torch.bool, device=td.device).unsqueeze(0)
        skip_mask = td["visited"].unsqueeze(1) | eye  # [B, n+1, n+1]
        grandsucc_check = (grandsucc_feasible | skip_mask).all(dim=-1)

        unvisited = ~td["visited"]
        can_visit = unvisited & succ_feasible & grandsucc_check  # [B, n+1]

    return can_visit


class TSPTWGreedy:
    def __init__(self, greedy_type="nearest", get_mask=True, look_ahead_step=1):
        self.get_mask = get_mask
        self.look_ahead_step = look_ahead_step
        self.round_error_epsilon = 1e-5
        self.greedy_type = greedy_type

    def _reset(self, td=None, batch_size=None):
        device = td.device
        num_locs = td["locs"].size(1)

        visited = torch.zeros((*batch_size, num_locs), dtype=torch.bool, device=device)
        visited[:, 0] = True

        distance_matrix = torch.cdist(td["locs"], td["locs"], p=2, compute_mode="donot_use_mm_for_euclid_dist")
        duration_matrix = distance_matrix + td["service_time"][:, :, None]

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "service_time": td["service_time"],
                "time_windows": td["time_windows"],
                "distance_matrix": distance_matrix,
                "duration_matrix": duration_matrix,
                "current_time": torch.zeros(*batch_size, dtype=torch.float32, device=device),
                "current_node": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "service_start_time_cache": torch.zeros_like(visited),
                "action": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "visited": visited,
                "done": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "time_window_violation": torch.zeros((*batch_size, num_locs), dtype=torch.float32, device=device),
            },
            batch_size=batch_size,
            device=device,
        )

        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_action_mask(self, td):
        unvisited = ~td["visited"]
        if self.get_mask:
            can_visit = get_tsptw_action_mask(td, look_ahead_step=self.look_ahead_step, round_error_epsilon=self.round_error_epsilon)
            action_mask = torch.where(can_visit.any(-1, keepdim=True), can_visit, unvisited)
        else:
            action_mask = unvisited
        return action_mask

    def select_action(self, td):
        batch_idx = torch.arange(td["locs"].size(0), device=td.device)
        current_node = td["current_node"]
        action_mask = td["action_mask"]  # [B, n+1]

        if self.greedy_type == "nearest":
            d_ij = td["distance_matrix"][batch_idx, current_node, :]
            d_ij = d_ij.masked_fill(~action_mask, float("inf"))  # [B, n+1]
            action = d_ij.argmin(dim=-1)  # [B,]

        elif self.greedy_type == "min_resource":
            tw_late = td["time_windows"][..., 1]
            tw_late = tw_late.masked_fill(~action_mask, float("inf"))
            action = tw_late.argmin(dim=-1)
        else:  # random
            probs = (~td["action_mask"]).float()
            probs = probs / probs.sum(dim=-1, keepdim=True)
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return action

    def _step(self, td):
        """
        update the state of the environment, including
        current_node, current_time, time_window_violation, visited and action_mask
        """
        batch_idx = torch.arange(td["locs"].size(0), device=td.device)
        prev_node, curr_node = td["current_node"], td["action"]
        prev_loc, curr_loc = (
            td["locs"][batch_idx, prev_node],
            td["locs"][batch_idx, curr_node],
        )  # [B, 2]

        travel_time = get_distance(prev_loc, curr_loc)  # [B,]

        arrival_time = td["current_time"] + travel_time
        tw_early_curr, tw_late_curr = (td["time_windows"][batch_idx, curr_node]).unbind(-1)
        service_time = td["service_time"][batch_idx, curr_node]
        curr_time = torch.max(arrival_time, tw_early_curr) + service_time
        td["time_window_violation"][batch_idx, curr_node] = torch.clamp(arrival_time - tw_late_curr, min=0.0)

        visited = td["visited"].scatter_(1, curr_node[..., None], True)
        done = visited.sum(dim=-1) == visited.size(-1)
        reward = torch.zeros_like(done, dtype=torch.float32)

        td.update(
            {
                "current_time": curr_time,
                "current_node": curr_node,
                "visited": visited,
                "done": done,
                "reward": reward,
            }
        )
        num_unvisited = (~td["visited"][0]).sum().item()
        action_mask = self.get_action_mask(td) if num_unvisited > 1 else ~visited

        td.set("action_mask", action_mask)

        return td

    def rollout(self, td, device="cuda"):
        with torch.inference_mode():
            with torch.amp.autocast("cuda") if "cuda" in str(device) else torch.inference_mode():
                td = td.to(device)
                td = self._reset(td, batch_size=td.batch_size)
                actions = []
                while not td["done"].all():
                    td["action"] = self.select_action(td)
                    td = self._step(td)
                    actions.append(td["action"])
        actions = torch.stack(actions, dim=1)  # [B, n]
        reward_td = self._get_reward(td, actions)
        sol_feas = reward_td["total_constraint_violation"] < self.round_error_epsilon
        reward = reward_td["negative_length"]

        return TensorDict(
            {"actions": actions, "reward": reward, "sol_feas": sol_feas},
            batch_size=td.batch_size[0],
            device=td.device,
        )

    def _get_reward(self, td, actions):
        tour_length = get_tour_length(td, actions)

        tw_viol = td["time_window_violation"]  # [B, n+1]
        total_constraint_violation = tw_viol.sum(dim=1)  # [B]
        violated_node_count = (tw_viol > self.round_error_epsilon).sum(dim=1).float()  # [B]

        return TensorDict(
            {
                "negative_length": -tour_length,
                "total_constraint_violation": total_constraint_violation,
                "violated_node_count": violated_node_count,
            },
            batch_size=td["locs"].size(0),
            device=td.device,
        )

if __name__ == "__main__":
    import argparse
    from baselines.greedy_driver import greedy_solver
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2500)
    parser.add_argument("--data_dir", type=str, default="data/random")
    parser.add_argument("--problem_size", type=int, default=50, help="Problem size")
    parser.add_argument("--hardness", type=str, choices=["easy", "medium", "hard"], default="hard", help="Problem difficulty")
    parser.add_argument("--greedy_type", type=str, default="nearest")
    parser.add_argument("--get_mask", type=bool, default=True)

    args = parser.parse_args()
    data_dir, problem, problem_size, hardness = args.data_dir, "tsptw", args.problem_size, args.hardness

    test_path = f"{data_dir}/{problem}/test/{problem}{problem_size}_test_{hardness}_seed2025.npz"
    test_dir = os.path.dirname(test_path)
    reference_solver = "pyvrp" if problem == "tsptw" else "lkh"
    ref_sol_path = os.path.join(test_dir, f"{reference_solver}_{problem_size}_{hardness}.npz")

    greedy_solver(
        problem_name=problem,
        test_path=test_path,
        ref_sol_path=ref_sol_path,
        batch_size=args.batch_size,
        greedy_type=args.greedy_type,
        look_ahead_step=2,
    )
