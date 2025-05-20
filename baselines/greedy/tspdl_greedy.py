import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)
import torch
from tensordict.tensordict import TensorDict
from rl4co.utils.ops import get_distance, gather_by_index

def get_action_mask(td, look_ahead_step=2, round_error_epsilon=1e-5):
    if look_ahead_step == 1:
        load_on_arrival = td["current_load"].unsqueeze(-1) + td["demand"]
        meets_draft_limit = load_on_arrival <= (td["draft_limit"] + round_error_epsilon)
        unvisited = ~td["visited"]
        can_visit_local = unvisited & meets_draft_limit

        any_draft_limit_viol = (unvisited & ~meets_draft_limit).any(dim=-1, keepdim=True)  # [B, 1]
        can_visit = torch.where(any_draft_limit_viol, torch.zeros_like(can_visit_local), can_visit_local)
    elif look_ahead_step == 2:
        load_succ = td["current_load"].unsqueeze(-1) + td["demand"]  # [B, n+1]
        load_grandsucc = load_succ.unsqueeze(-1) + td["demand"].unsqueeze(1)  # [B, n+1, n+1]

        succ_feasible = load_succ <= (td["draft_limit"] + round_error_epsilon)  # [B, n+1]
        grandsucc_feasible = load_grandsucc <= (td["draft_limit"].unsqueeze(1) + round_error_epsilon)  # [B, n+1, n+1]

        eye = torch.eye(td["locs"].size(1), dtype=torch.bool, device=td.device).unsqueeze(0)
        skip_mask = td["visited"].unsqueeze(1) | eye  # [B, n+1, n+1]
        grandsucc_check = (grandsucc_feasible | skip_mask).all(dim=-1)

        unvisited = ~td["visited"]
        can_visit = unvisited & succ_feasible & grandsucc_check  # [B, n+1]

    return can_visit

class TSPDLGreedy:
    def __init__(self, greedy_type="nearest", get_mask=True, look_ahead_step=2):
        self.get_mask = get_mask
        self.look_ahead_step = look_ahead_step
        self.round_error_epsilon = 1e-5
        self.greedy_type = greedy_type
    
    def _reset(self, td=None, batch_size=None):
        visited = torch.zeros((*batch_size, td["locs"].size(1)), dtype=torch.bool, device=td.device)
        visited[:, 0] = True

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "demand": td["demand"],
                "draft_limit": td["draft_limit"],
                "current_node": torch.zeros(*batch_size, dtype=torch.int64, device=td.device),
                "current_load": torch.zeros(*batch_size, dtype=torch.float32, device=td.device),
                "draft_limit_violation": torch.zeros_like(visited, dtype=torch.float32, device=td.device),
                "done": torch.zeros(*batch_size, dtype=torch.bool, device=td.device),
                "visited": visited,
            },
            batch_size=td.batch_size,
            device=td.device,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_action_mask(self, td):
        unvisited = ~td["visited"]
        if self.get_mask:
            can_visit = get_action_mask(td, look_ahead_step=self.look_ahead_step, round_error_epsilon=self.round_error_epsilon)
            action_mask = torch.where(can_visit.any(-1, keepdim=True), can_visit, unvisited)
        else:
            action_mask = unvisited
        return action_mask
    
    def _step(self, td):
        batch_idx = torch.arange(td["locs"].size(0), device=td.device)
        current_node = td["action"]
        current_load = td["current_load"] + gather_by_index(td["demand"], current_node)
        current_draft_limit = gather_by_index(td["draft_limit"], current_node)
        td["draft_limit_violation"][batch_idx, current_node] = (current_load - current_draft_limit).clamp_(min=0.0)

        visited = td["visited"].scatter_(1, current_node.unsqueeze(1), 1)
        done = visited.sum(1) == visited.size(1)
        reward = torch.zeros_like(done, dtype=torch.float32)
        td.update(
            {
                "visited": visited,
                "current_node": current_node,
                "current_load": current_load,
                "reward": reward,
                "done": done,
            }
        )
        num_unvisited = (~td["visited"][0]).sum().item()
        action_mask = self.get_action_mask(td) if num_unvisited > 1 else ~visited
        td.set("action_mask", action_mask)
        return td
    
    def select_action(self, td):
        action_mask = td["action_mask"]  # [B, n+1]

        if self.greedy_type == "nearest":
            d_ij = get_distance(gather_by_index(td["locs"], td["current_node"])[:, None, :], td["locs"]) # [B, n+1]
            d_ij = d_ij.masked_fill(~action_mask, float("inf"))  # [B, n+1]
            action = d_ij.argmin(dim=-1)  # [B,]

        elif self.greedy_type == "min_resource":
            draft_limit = td["draft_limit"].masked_fill(~action_mask, float("inf"))  
            action = draft_limit.argmin(dim=-1)  
        else:  # random
            probs = (~td["action_mask"]).float()
            probs = probs / probs.sum(dim=-1, keepdim=True)
            action = torch.multinomial(probs, num_samples=1).squeeze(-1) 

        return action
    
    def _get_reward(self, td, actions):

        ordered_locs = torch.cat([td["locs"][:, 0:1], gather_by_index(td["locs"], actions)], dim=1)
        diff = ordered_locs - ordered_locs.roll(-1, dims=-2)
        tour_length = diff.norm(dim=-1).sum(-1)

        draft_limit_viol = td["draft_limit_violation"]  # [B, n+1]
        total_constraint_violation = draft_limit_viol.sum(dim=1)  # [B]
        violated_node_count = (draft_limit_viol > self.round_error_epsilon).sum(dim=1).float()  # [B]

        return TensorDict(
            {
                "negative_length": -tour_length,
                "total_constraint_violation": total_constraint_violation,
                "violated_node_count": violated_node_count,
            },
            batch_size=td["locs"].size(0),
            device=td.device,
        )

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
            {"actions": actions,
             "reward": reward,
             "sol_feas": sol_feas},
             batch_size= td.batch_size[0],
             device=td.device,
        )
    

if __name__ == "__main__":
    import argparse
    from baselines.greedy_driver import greedy_solver
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2500)
    parser.add_argument("--data_dir", type=str, default="data/random")
    parser.add_argument("--problem", type=str, choices=["tspdl", "tsptw"], default="tspdl", help="Problem type")
    parser.add_argument("--problem_size", type=int, choices=[50, 100], default=50, help="Problem size")
    parser.add_argument("--hardness", type=str, choices=["easy", "medium", "hard"], default="hard", help="Problem difficulty")
    parser.add_argument("--greedy_type", type=str, default="nearest")
    parser.add_argument("--get_mask", type=bool, default=True)
    parser.add_argument("--look_ahead_step", type=int, default=2)

    args = parser.parse_args()
    data_dir, problem, problem_size, hardness = args.data_dir, args.problem, args.problem_size, args.hardness

    test_path = f"{data_dir}/{problem}/test/{problem}{problem_size}_test_{hardness}_seed2025.npz"
    test_dir = os.path.dirname(test_path)
    reference_solver = "pyvrp" if problem == "tsptw" else "lkh"
    ref_sol_path = os.path.join(test_dir, f"{reference_solver}_{problem_size}_{hardness}.npz")

    greedy_solver(
        problem_name=args.problem,
        test_path=test_path,
        ref_sol_path=ref_sol_path,
        batch_size=args.batch_size,
        greedy_type=args.greedy_type,
        look_ahead_step=args.look_ahead_step,
    )