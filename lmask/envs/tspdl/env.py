from tqdm import tqdm
import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from rl4co.utils.ops import gather_by_index, batchify, unbatchify
from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from lmask.envs.tspdl.generator import TSPDLGenerator
from ...utils.ops import get_action_from_logits_and_mask, slice_cache


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


class TSPDLEnv(RL4COEnvBase):
    def __init__(self, generator=TSPDLGenerator, generator_params={}, **kwargs):
        self.look_ahead_step = kwargs.pop("look_ahead_step", 2)
        kwargs.pop("max_backtrack_steps", None)
        super().__init__(check_solution=False, **kwargs)
        self.generator = generator(**generator_params)
        self.round_error_epsilon = 1e-5

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
                "draft_limit_violation": torch.zeros_like(visited, dtype=torch.float32),
                "visited": visited,
            },
            batch_size=td.batch_size,
            device=td.device,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_action_mask(self, td):
        unvisited = ~td["visited"]
        can_visit = get_action_mask(td, look_ahead_step=self.look_ahead_step, round_error_epsilon=self.round_error_epsilon)  # [B, n+1]
        action_mask = torch.where(can_visit.any(-1, keepdim=True), can_visit, unvisited)
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


class TSPDLLazyMaskEnv(RL4COEnvBase):
    def __init__(self, generator=TSPDLGenerator, generator_params={}, **kwargs):
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
                "demand": td["demand"],
                "draft_limit": td["draft_limit"],
                "current_load": torch.zeros(*batch_size, dtype=torch.float32, device=device),
                "current_node": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "visited": visited,
                "backtrack_steps": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "step_idx": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "load_stack": torch.zeros((*batch_size, num_locs), dtype=torch.float32, device=device),
                "node_stack": torch.zeros((*batch_size, num_locs), dtype=torch.int64, device=device),
                "terminated": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "truncated": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "done": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "action": torch.zeros(*batch_size, dtype=torch.int64, device=device),
            },
            batch_size=batch_size,
            device=td.device,
        )
        action_mask = self.get_action_mask(td_reset)
        mask_stack = torch.zeros((*batch_size, num_locs, num_locs), dtype=torch.bool, device=device)
        mask_stack[:, 0] = action_mask
        td.update({"action_mask": action_mask, "mask_stack": mask_stack})
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
        new_node = td["action"]
        new_load = td["current_load"] + td["demand"][batch_idx, new_node]

        visited = td["visited"].scatter_(1, new_node[..., None], True)

        new_step_idx = td["step_idx"] + 1
        td["done"] = visited.all(dim=-1)

        td["reward"] = torch.zeros_like(td["done"], dtype=torch.float32)
        td["load_stack"][batch_idx, new_step_idx] = new_load
        td["node_stack"][batch_idx, new_step_idx] = new_node
        td.update(
            {
                "step_idx": new_step_idx,
                "visited": visited,
                "current_load": new_load,
                "current_node": new_node,
            }
        )

        action_mask = self.get_action_mask(td)
        td.set("action_mask", action_mask)
        td["mask_stack"][batch_idx, new_step_idx] = action_mask
        return td

    def backtrack(self, td):
        step_idx = td["step_idx"]
        # Assert all instances are valid for backtracking
        assert (step_idx - 1 >= 0).all(), "Some instances are at the depot node and cannot backtrack further, indicating no feasible solution exists"

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
                "current_load": td["load_stack"][batch_idx, new_step_idx],
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

            step_foward_mask = active & has_feas
            backtrack_mask = active & (~has_feas)

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
