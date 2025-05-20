from tqdm import tqdm
import torch
from tensordict.tensordict import TensorDict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.decoding import get_log_likelihood
from rl4co.utils.ops import get_distance, gather_by_index, batchify, unbatchify, calculate_entropy
from rl4co.data.transforms import StateAugmentation
from .generator import TSPTWGenerator
from ...utils.ops import get_action_from_logits_and_mask, get_tour_length, get_time_window_violations, slice_cache


def get_action_mask(td, look_ahead_step=2, round_error_epsilon=1e-5):
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


class TSPTWEnv(RL4COEnvBase):
    def __init__(self, generator=TSPTWGenerator, generator_params={}, **kwargs):
        self.look_ahead_step = kwargs.pop("look_ahead_step", 2)
        [kwargs.pop(k, None) for k in ("max_backtrack_steps", "local_search", "num_workers", "max_trials")]
        super().__init__(check_solution=False, **kwargs)
        self.generator = generator(**generator_params)
        self.round_error_epsilon = 1e-5

    def _reset(self, td=None, batch_size=None):
        device = td.device
        num_locs = td["locs"].size(1)

        visited = torch.zeros((*batch_size, num_locs), dtype=torch.bool, device=device)
        visited[:, 0] = True

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "service_time": td["service_time"],
                "time_windows": td["time_windows"],
                "current_time": torch.zeros(*batch_size, dtype=torch.float32, device=device),
                "current_node": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "service_start_time_cache": torch.zeros_like(visited),
                "action": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "visited": visited,
                "time_window_violation": torch.zeros((*batch_size, num_locs), dtype=torch.float32, device=device),
            },
            batch_size=batch_size,
            device=device,
        )
        if self.look_ahead_step == 2:
            td_reset["distance_matrix"] = torch.cdist(td["locs"], td["locs"], p=2, compute_mode="donot_use_mm_for_euclid_dist")
            td_reset["duration_matrix"] = td_reset["distance_matrix"] + td["service_time"][:, :, None]
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_action_mask(self, td):
        if self.look_ahead_step == 1:
            return ~td["visited"]
        elif self.look_ahead_step == 2:
            unvisited = ~td["visited"]
            can_visit = get_action_mask(td, look_ahead_step=self.look_ahead_step, round_error_epsilon=self.round_error_epsilon)
            action_mask = torch.where(can_visit.any(-1, keepdim=True), can_visit, unvisited)
        return action_mask

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

    def rollout(self, td, policy, num_samples=1, decode_type="greedy", device="cuda", **kwargs):
        with torch.inference_mode():
            with torch.amp.autocast("cuda") if "cuda" in str(device) else torch.inference_mode():
                td = td.to(device)
                td = self.reset(td)
                td_aug = StateAugmentation(num_augment=8, augment_fn="dihedral8")(td)
                num_samples = 0 if decode_type == "greedy" else num_samples

                out = policy(td_aug, self, decode_type=decode_type, num_samples=num_samples)
                actions = unbatchify(out["actions"], (8, num_samples))

                reward_td = unbatchify(out["reward"], (8, num_samples))
                reward, total_constraint_violation = (
                    reward_td["negative_length"],
                    reward_td["total_constraint_violation"],
                )
                sol_feas = total_constraint_violation < self.round_error_epsilon

        return TensorDict(
            {
                "actions": actions,
                "reward": reward,
                "sol_feas": sol_feas,
            },
            batch_size=td.batch_size[0],
            device=td.device,
        )

    def get_penalized_reward(self, td, actions, rho_c=1.0, rho_n=1.0, return_dict=False):
        tour_length = get_tour_length(td, actions)
        tw_viol = get_time_window_violations(td, actions)
        total_constraint_violation = tw_viol.sum(dim=1)
        violated_node_count = (tw_viol > self.round_error_epsilon).sum(dim=1).float()
        penalized_obj = tour_length + rho_c * total_constraint_violation + rho_n * violated_node_count

        if return_dict:
            return TensorDict(
                {
                    "negative_length": -tour_length,
                    "total_constraint_violation": total_constraint_violation,
                    "violated_node_count": violated_node_count,
                },
                batch_size=td["locs"].size(0),
                device=td.device,
            )
        return -penalized_obj


class TSPTWLazyMaskEnv(RL4COEnvBase):
    def __init__(self, generator=TSPTWGenerator, generator_params={}, **kwargs):
        self.max_backtrack_steps = kwargs.pop("max_backtrack_steps", 300)
        self.look_ahead_step = kwargs.pop("look_ahead_step", 2)
        super().__init__(check_solution=False, **kwargs)
        self.generator = generator(**generator_params)
        self.round_error_epsilon = 1e-5

    def _reset(self, td=None, batch_size=None):
        device = td.device
        num_locs = td["locs"].size(1)
        visited = torch.zeros((*batch_size, num_locs), dtype=torch.bool, device=device)
        visited[:, 0] = True

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "service_time": td["service_time"],
                "time_windows": td["time_windows"],
                "current_time": torch.zeros(*batch_size, dtype=torch.float32, device=device),
                "current_node": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "visited": visited,
                "backtrack_steps": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "step_idx": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "time_stack": torch.zeros((*batch_size, num_locs), dtype=torch.float32, device=device),
                "node_stack": torch.zeros((*batch_size, num_locs), dtype=torch.int64, device=device),
                "terminated": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "truncated": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "done": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "action": torch.zeros(*batch_size, dtype=torch.int64, device=device),
            },
            batch_size=batch_size,
            device=device,
        )
        if self.look_ahead_step == 2:
            td_reset["duration_matrix"] = torch.cdist(td["locs"], td["locs"], p=2, compute_mode="donot_use_mm_for_euclid_dist")
        action_mask = self.get_action_mask(td_reset)
        mask_stack = torch.zeros((*batch_size, num_locs, num_locs), dtype=torch.bool, device=device)
        mask_stack[:, 0] = action_mask
        td_reset.update({"action_mask": action_mask, "mask_stack": mask_stack})
        return td_reset

    def get_action_mask(self, td):
        return get_action_mask(td, look_ahead_step=self.look_ahead_step, round_error_epsilon=self.round_error_epsilon)

    def _get_reward(self, td, actions):

        ordered_locs = torch.cat([td["locs"][:, 0:1], gather_by_index(td["locs"], actions)], dim=1)
        diff = ordered_locs - ordered_locs.roll(-1, dims=-2)
        tour_length = diff.norm(dim=-1).sum(-1)

        return -tour_length

    def _step(self, td):

        batch_idx = torch.arange(td["locs"].size(0), device=td.device)
        curr_node, new_node = td["current_node"], td["action"]
        curr_loc, new_loc = gather_by_index(td["locs"], curr_node), gather_by_index(td["locs"], new_node)

        travel_time = get_distance(curr_loc, new_loc)
        arrival_time = td["current_time"] + travel_time
        tw_early_new = gather_by_index(td["time_windows"][..., 0], new_node)
        service_time = gather_by_index(td["service_time"], new_node)
        new_time = torch.max(arrival_time, tw_early_new) + service_time

        visited = td["visited"].scatter_(1, new_node[..., None], True)

        new_step_idx = td["step_idx"] + 1
        td["done"] = visited.all(dim=-1)

        td["reward"] = torch.zeros_like(td["done"], dtype=torch.float32)
        td["time_stack"][batch_idx, new_step_idx] = new_time
        td["node_stack"][batch_idx, new_step_idx] = new_node
        td.update(
            {
                "step_idx": new_step_idx,
                "visited": visited,
                "current_time": new_time,
                "current_node": new_node,
            }
        )
        action_mask = self.get_action_mask(td)
        td.set("action_mask", action_mask)
        td["mask_stack"][batch_idx, new_step_idx] = action_mask
        return td

    def backtrack(self, td):
        step_idx = td["step_idx"]
        new_step_idx = step_idx - 1

        batch_idx = torch.arange(td["locs"].size(0), device=td.device)
        deleted_node = td["node_stack"][batch_idx, step_idx]
        td["visited"].scatter_(1, deleted_node[..., None], False)

        td["backtrack_steps"] += 1
        td["truncated"] = td["backtrack_steps"] >= self.max_backtrack_steps

        td["mask_stack"][batch_idx, new_step_idx, deleted_node] = False

        td.update(
            {
                "done": td["truncated"],
                "step_idx": new_step_idx,
                "action_mask": td["mask_stack"][batch_idx, new_step_idx],  # [B, N+1]
                "current_time": td["time_stack"][batch_idx, new_step_idx],
                "current_node": td["node_stack"][batch_idx, new_step_idx],
            }
        )
        return td

    def decode_multi(self, td, hidden, policy, num_samples=1, decode_type="greedy"):
        batch_size = td["locs"].size(0)
        pbar = tqdm(total=batch_size, desc="Number of finished instances")
        prev_done = 0
        while not td["done"].all():
            curr_done = td["done"].sum().item()
            if curr_done > prev_done:
                pbar.update(curr_done - prev_done)
                prev_done = curr_done

            logits, mask = policy.decoder(td, hidden, num_samples)  # [B*S, n+1]

            active = ~td["done"]
            has_feas = td["action_mask"].any(dim=-1)
            step_foward_mask = active & has_feas
            backtrack_mask = active & (~has_feas)

            if step_foward_mask.any():
                step_forward_idx = step_foward_mask.nonzero(as_tuple=True)[0]
                valid_logits, valid_mask = logits[step_forward_idx], mask[step_forward_idx]

                td["action"][step_forward_idx] = get_action_from_logits_and_mask(valid_logits, valid_mask, decode_type=decode_type)
                td[step_forward_idx] = self._step(td[step_forward_idx])

            if backtrack_mask.any():
                backtrack_idx = backtrack_mask.nonzero(as_tuple=True)[0]
                td[backtrack_idx] = self.backtrack(td[backtrack_idx])

        pbar.update(batch_size - prev_done)
        pbar.close()
        return td

    def decode_fast(self, td, hidden, policy, num_samples=1, decode_type="greedy"):
        batch_size = td["locs"].size(0)
        assert num_samples <= 1, "This fast implmentation of lazymask decoding only supports num_samples=1"
        pbar = tqdm(total=batch_size, desc="Number of finished instances")
        prev_done = 0
        while not td["done"].all():
            curr_done = td["done"].sum().item()
            if curr_done > prev_done:
                pbar.update(curr_done - prev_done)
                prev_done = curr_done

            active = ~td["done"]
            has_feas = td["action_mask"].any(dim=-1)
            at_depot = td["step_idx"] == 0

            step_foward_mask = active & has_feas
            backtrack_mask = active & (~has_feas) & (~at_depot)
            invalid_mask = active & (~has_feas) & (at_depot)

            if step_foward_mask.any():
                step_forward_idx = step_foward_mask.nonzero(as_tuple=True)[0]
                td_step = td[step_forward_idx]

                # Note: This slicing approach currently only works for single sample scenarios (not num_samples > 1)
                # When S > 1, the network internally needs to unbatchify a [B*S] tensor into [B, S].
                # Taking a subtensor has two issues:
                # 1. The size of the subtensor may not be a multiple of S
                # 2. Even if it is a multiple of S, the ordering is disrupted, so unbatchifying into [\tilde{B}, S]
                #    may result in mixing solutions from different instances in the same S dimension, causing confusion
                # TODO: Generalize it to support multi-sample scenarios
                cache_step = slice_cache(hidden, step_forward_idx)
                logits, mask = policy.decoder(td_step, cache_step)

                td["action"][step_forward_idx] = get_action_from_logits_and_mask(logits, mask, decode_type=decode_type)
                td[step_forward_idx] = self._step(td[step_forward_idx])

            if backtrack_mask.any():
                backtrack_idx = backtrack_mask.nonzero(as_tuple=True)[0]
                td[backtrack_idx] = self.backtrack(td[backtrack_idx])

            if invalid_mask.any():
                invalid_idx = invalid_mask.nonzero(as_tuple=True)[0]
                td["done"][invalid_idx] = True
                td["terminated"][invalid_idx] = True
        pbar.update(batch_size - prev_done)
        pbar.close()

        return td

    def rollout(self, td, policy, num_samples=1, num_augment=8, decode_type="greedy", device="cuda", return_td=False, **kwargs):
        with torch.inference_mode():
            with torch.amp.autocast("cuda") if "cuda" in str(device) else torch.inference_mode():
                td = td.to(device)
                td = self.reset(td)
                if num_augment > 1:
                    td = StateAugmentation(num_augment=8, augment_fn="dihedral8")(td)
                td["done"].squeeze_(-1)
                td["terminated"].squeeze_(-1)

                hidden, _ = policy.encoder(td)
                num_samples = 0 if decode_type == "greedy" else num_samples
                if num_samples > 1 and decode_type == "sampling":
                    td = batchify(td, num_samples)
                td, _, hidden = policy.decoder.pre_decoder_hook(td, self, hidden, num_samples)

                if num_samples <= 1:
                    td = self.decode_fast(td, hidden, policy, num_samples, decode_type)
                else:
                    td = self.decode_multi(td, hidden, policy, num_samples, decode_type)

                out = self.get_rollout_result(td)
                if num_augment > 1:
                    out = unbatchify(out, (8, num_samples))
        return (out, td) if return_td else out

    def get_rollout_result(self, td):
        actions = td["node_stack"][:, 1:]
        negative_length = self._get_reward(td, actions)
        sol_feas = td["step_idx"] == (td["locs"].size(1) - 1)
        return TensorDict(
            {
                "actions": actions,
                "reward": negative_length,
                "sol_feas": sol_feas,
            },
            batch_size=td.batch_size[0],
            device=td.device,
        )


class TSPTWLazyMaskTrainEnv(RL4COEnvBase):
    def __init__(self, generator=TSPTWGenerator, generator_params={}, **kwargs):
        self.max_backtrack_steps = kwargs.pop("max_backtrack_steps", 300)
        self.look_ahead_step = kwargs.pop("look_ahead_step", 2)
        super().__init__(check_solution=False, **kwargs)
        self.generator = generator(**generator_params)
        self.round_error_epsilon = 1e-5

    def _reset(self, td=None, batch_size=None):
        device = td.device
        num_locs = td["locs"].size(1)
        visited = torch.zeros((*batch_size, num_locs), dtype=torch.bool, device=device)
        visited[:, 0] = True

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "service_time": td["service_time"],
                "time_windows": td["time_windows"],
                "current_time": torch.zeros(*batch_size, dtype=torch.float32, device=device),
                "current_node": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "visited": visited,
                "backtrack_steps": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "backtrack_budget_reached": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "confirmed_infeasible": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "step_idx": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "revisit_count_stack": torch.zeros((*batch_size, num_locs), dtype=torch.int64, device=device),
                "time_stack": torch.zeros((*batch_size, num_locs), dtype=torch.float32, device=device),
                "node_stack": torch.zeros((*batch_size, num_locs), dtype=torch.int64, device=device),
                "mask_stack": torch.zeros((*batch_size, num_locs, num_locs), dtype=torch.bool, device=device),
                "finished": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "action": torch.zeros(*batch_size, dtype=torch.int64, device=device),
            },
            batch_size=batch_size,
            device=device,
        )
        if self.look_ahead_step == 2:
            td_reset["duration_matrix"] = torch.cdist(td["locs"], td["locs"], p=2, compute_mode="donot_use_mm_for_euclid_dist")
            td_reset["distance_matrix"] = td_reset["duration_matrix"]
        td_reset["action_mask"] = self.get_action_mask(td_reset)
        td_reset["mask_stack"][:, 0] = td_reset["action_mask"]
        return td_reset

    def get_action_mask(self, td):
        can_visit = get_action_mask(td, look_ahead_step=self.look_ahead_step, round_error_epsilon=self.round_error_epsilon)  # [B, n+1]

        no_valid_action = ~can_visit.any(dim=-1, keepdim=True)
        budget_reached = td["backtrack_budget_reached"].unsqueeze(-1)  # [B, 1]

        # For those instances that have no valid action and have reached the budget,
        # we need to set the unvisited nodes as valid actions to allow them to step forward
        no_valid_and_budget_reached = no_valid_action & budget_reached
        unvisited = ~td["visited"]
        action_mask = torch.where(no_valid_and_budget_reached, unvisited, can_visit)

        confirmed_infeasible = no_valid_and_budget_reached & unvisited.any(dim=-1, keepdim=True)  # [B, 1]
        td["confirmed_infeasible"] = td["confirmed_infeasible"] | confirmed_infeasible.squeeze(-1)  # [B, ]

        return action_mask

    def _get_reward(self, td, actions):

        ordered_locs = torch.cat([td["locs"][:, 0:1], gather_by_index(td["locs"], actions)], dim=1)
        diff = ordered_locs - ordered_locs.roll(-1, dims=-2)
        tour_length = diff.norm(dim=-1).sum(-1)

        return -tour_length

    def _step(self, td):

        batch_idx = torch.arange(td["locs"].size(0), device=td.device)
        curr_node, new_node = td["current_node"], td["action"]
        curr_loc, new_loc = gather_by_index(td["locs"], curr_node), gather_by_index(td["locs"], new_node)

        travel_time = get_distance(curr_loc, new_loc)
        arrival_time = td["current_time"] + travel_time
        tw_early_new = gather_by_index(td["time_windows"][..., 0], new_node)
        service_time = gather_by_index(td["service_time"], new_node)
        new_time = torch.max(arrival_time, tw_early_new) + service_time

        visited = td["visited"].scatter_(1, new_node[..., None], True)

        new_step_idx = td["step_idx"] + 1
        td["finished"] = visited.all(dim=-1)

        td["reward"] = torch.zeros_like(td["finished"], dtype=torch.float32)
        td["time_stack"][batch_idx, new_step_idx] = new_time
        td["node_stack"][batch_idx, new_step_idx] = new_node
        # we confirm that for the steppinf forward instances, the new partial tour is not visited before.
        td["revisit_count_stack"][batch_idx, new_step_idx] = 0
        td.update(
            {
                "step_idx": new_step_idx,
                "visited": visited,
                "current_time": new_time,
                "current_node": new_node,
            }
        )
        td["action_mask"] = self.get_action_mask(td)
        td["mask_stack"][batch_idx, new_step_idx] = td["action_mask"]
        return td

    def backtrack(self, td):
        step_idx = td["step_idx"]
        new_step_idx = step_idx - 1
        assert (new_step_idx >= 0).all(), "step index for backtracking instances should be greater than or equal to 0"
        batch_idx = torch.arange(td["locs"].size(0), device=td.device)
        deleted_node = td["node_stack"][batch_idx, step_idx]
        td["visited"].scatter_(1, deleted_node[..., None], False)

        td["backtrack_steps"] += 1
        td["backtrack_budget_reached"] = td["backtrack_steps"] >= self.max_backtrack_steps

        td["mask_stack"][batch_idx, new_step_idx, deleted_node] = False
        td["revisit_count_stack"][batch_idx, new_step_idx] = td["revisit_count_stack"][batch_idx, new_step_idx] + 1
        td.update(
            {
                "step_idx": new_step_idx,
                "action_mask": td["mask_stack"][batch_idx, new_step_idx],  # [B, N+1]
                "current_time": td["time_stack"][batch_idx, new_step_idx],
                "current_node": td["node_stack"][batch_idx, new_step_idx],
            }
        )
        return td

    def decode_multi(self, td, hidden, policy, num_samples=1, decode_type="sampling"):
        batch_size, num_locs, _ = td["locs"].size()
        logprobs_stack = torch.zeros((batch_size, num_locs, num_locs), dtype=torch.float32, device=td.device, requires_grad=True)

        while not td["finished"].all():
            logits, mask = policy.decoder(td, hidden, num_samples)  # [B*S, n+1]

            active = ~td["finished"]
            has_feas = td["action_mask"].any(dim=-1)
            at_depot = td["step_idx"] == 0
            step_foward_mask = active & has_feas
            backtrack_mask = active & ~has_feas & (~at_depot)
            invalid_mask = active & ~has_feas & at_depot

            if step_foward_mask.any():
                step_forward_idx = step_foward_mask.nonzero(as_tuple=True)[0]
                valid_logits, valid_mask = logits[step_forward_idx], mask[step_forward_idx]

                logprobs, td["action"][step_forward_idx] = get_action_from_logits_and_mask(valid_logits, valid_mask, decode_type=decode_type, return_logprobs=True)
                td[step_forward_idx] = self._step(td[step_forward_idx])

                step_idx = td[step_forward_idx]["step_idx"]
                logprobs_stack = logprobs_stack.index_put(indices=(step_forward_idx, step_idx.clone()), values=logprobs, accumulate=False)

            if backtrack_mask.any():
                backtrack_idx = backtrack_mask.nonzero(as_tuple=True)[0]
                td[backtrack_idx] = self.backtrack(td[backtrack_idx])

            if invalid_mask.any():
                invalid_idx = invalid_mask.nonzero(as_tuple=True)[0]
                td["finished"][invalid_idx] = True
                td["terminated"][invalid_idx] = True

        return td, logprobs_stack

    def decode_fast(self, td, hidden, policy, num_samples=1, decode_type="sampling"):
        batch_size = td["locs"].size(0)
        batch_size, num_locs, _ = td["locs"].size()
        logprobs_stack = torch.zeros((batch_size, num_locs, num_locs), dtype=torch.float32, device=td.device, requires_grad=True)
        assert num_samples <= 1, "This fast implmentation of lazymask decoding only supports num_samples=1"
        pbar = tqdm(total=batch_size, desc="Number of finished instances")
        prev_done = 0
        assert num_samples <= 1, "This fast implmentation of lazymask decoding only supports num_samples=1"

        while not td["finished"].all():
            curr_done = td["done"].sum().item()
            if curr_done > prev_done:
                pbar.update(curr_done - prev_done)
                prev_done = curr_done

            step_idx = td["step_idx"]
            active = ~td["finished"]
            has_feas = td["action_mask"].any(dim=-1)
            at_depot = td["step_idx"] == 0
            step_foward_mask = active & has_feas
            backtrack_mask = active & ~has_feas & (~at_depot)
            invalid_mask = active & ~has_feas & at_depot

            if step_foward_mask.any():
                step_forward_idx = step_foward_mask.nonzero(as_tuple=True)[0]
                td_step = td[step_forward_idx]

                # Note: This slicing approach currently only works for single sample scenarios (not num_samples > 1)
                # When S > 1, the network internally needs to unbatchify a [B*S] tensor into [B, S].
                # Taking a subtensor has two issues:
                # 1. The size of the subtensor may not be a multiple of S
                # 2. Even if it is a multiple of S, the ordering is disrupted, so unbatchifying into [\tilde{B}, S]
                #    may result in mixing solutions from different instances in the same S dimension, causing confusion
                # TODO: Generalize it to support multi-sample scenarios
                cache_step = slice_cache(hidden, step_forward_idx)
                logits, mask = policy.decoder(td_step, cache_step)

                logprobs, td["action"][step_forward_idx] = get_action_from_logits_and_mask(logits, mask, decode_type=decode_type, return_logprobs=True)
                td[step_forward_idx] = self._step(td[step_forward_idx])

                step_idx = td[step_forward_idx]["step_idx"]
                logprobs_stack = logprobs_stack.index_put(indices=(step_forward_idx, step_idx), values=logprobs, accumulate=False)

            if backtrack_mask.any():
                backtrack_idx = backtrack_mask.nonzero(as_tuple=True)[0]
                td[backtrack_idx] = self.backtrack(td[backtrack_idx])

            if invalid_mask.any():
                invalid_idx = invalid_mask.nonzero(as_tuple=True)[0]
                td["finished"][invalid_idx] = True
                td["terminated"][invalid_idx] = True

        pbar.update(batch_size - prev_done)
        pbar.close()
        return td, logprobs_stack

    def train_rollout(self, td, policy, num_samples=1, return_td=False):
        hidden, _ = policy.encoder(td)
        if num_samples > 1:
            td = batchify(td, num_samples)
        td, _, hidden = policy.decoder.pre_decoder_hook(td, self, hidden, num_samples)

        if num_samples <= 1:
            td, logprobs_stack = self.decode_fast(td, hidden, policy, num_samples, decode_type="sampling")
        else:
            td, logprobs_stack = self.decode_multi(td, hidden, policy, num_samples, decode_type="sampling")
        out = self.get_rollout_result(td)
        out["log_likelihood"] = get_log_likelihood(logprobs_stack, td["node_stack"])
        out["entropy"] = calculate_entropy(logprobs_stack)
        return (out, td) if return_td else out

    def rollout(self, td_ori, policy, num_samples=1, num_augment=8, decode_type="greedy", device="cuda", return_td=False, **kwargs):
        with torch.no_grad():
            with torch.amp.autocast("cuda") if "cuda" in str(device) else torch.no_grad():
                td_ori = td_ori.to(device)
                td_ori = self.reset(td_ori)
                if num_augment > 1:
                    td = StateAugmentation(num_augment=8, augment_fn="dihedral8")(td_ori)

                hidden, _ = policy.encoder(td)
                num_samples = 0 if decode_type == "greedy" else num_samples
                if num_samples > 1 and decode_type == "sampling":
                    td = batchify(td, num_samples)
                td, _, hidden = policy.decoder.pre_decoder_hook(td, self, hidden, num_samples)

                if num_samples <= 1:
                    td, _ = self.decode_fast(td, hidden, policy, num_samples, decode_type)
                else:
                    td, _ = self.decode_multi(td, hidden, policy, num_samples, decode_type)
                out = self.get_rollout_result(td)

        return (out, td) if return_td else unbatchify(out, (8, num_samples))

    def get_rollout_result(self, td):
        actions = td["node_stack"][:, 1:]
        negative_length = self._get_reward(td, actions)

        tw_late = td["time_windows"][..., 1]
        tw_late_tour = gather_by_index(tw_late, actions)
        tw_viol = torch.clamp(td["time_stack"][:, 1:] - tw_late_tour, min=0.0)  # [B, n]

        total_constraint_violation = tw_viol.sum(dim=-1)
        violated_node_count = (tw_viol > self.round_error_epsilon).float().sum(dim=-1)
        sol_feas = total_constraint_violation < self.round_error_epsilon

        return TensorDict(
            {
                "reward": negative_length,
                "actions": actions,
                "negative_length": negative_length,
                "total_constraint_violation": total_constraint_violation,
                "violated_node_count": violated_node_count,
                "sol_feas": sol_feas,
            },
            batch_size=td.batch_size,
            device=td.device,
        )
