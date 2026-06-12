#!/usr/bin/env python3
"""Prototype controller for dynamic quantized-clone speculation windows.

This module is intentionally small and vLLM-version-aware. It can be used in
two ways:

1. Offline: parse benchmark logs and choose the best gamma by measured goodput.
2. Runtime prototype: monkeypatch a vLLM 0.6.x SpecDecodeWorker so a model
   loaded with max gamma can run each decode step with a smaller adaptive gamma.

The runtime hook is a research prototype, not a production patch. In vLLM 0.6.x,
turning speculation fully off for active sequences is not clean because the
draft KV cache becomes stale if the proposer is skipped. This file therefore
separates two cases:

* Dynamic gamma in 1..max_gamma is safe within active requests.
* Hard off (gamma 0) is safe as an admission/request-level choice, or for
  experiments where active requests will not be re-enabled before finishing.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Iterable


def _steady_tok_s(row: dict[str, Any]) -> float:
    return float(row.get("output_tok_s_steady_mean", row["output_tok_s_mean"]))


def iter_aggregate_rows(paths: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    for path_like in paths:
        path = Path(path_like)
        with path.open(errors="ignore") as handle:
            for line in handle:
                if line.startswith("RESULTS_AGG_JSON "):
                    row = json.loads(line.split(" ", 1)[1])
                    row["log_path"] = str(path)
                    yield row


def summarize_best_actions(
    paths: Iterable[str | Path],
    min_gain: float = 0.03,
) -> list[dict[str, Any]]:
    """Return best measured action per batch from benchmark aggregate logs."""
    target_by_batch: dict[int, dict[str, Any]] = {}
    spec_by_batch_gamma: dict[tuple[int, int], dict[str, Any]] = {}

    for row in iter_aggregate_rows(paths):
        batch = int(row["num_prompts"])
        if row["condition"] == "target":
            target_by_batch[batch] = row
        elif row["condition"].startswith("spec") and row.get("spec_tokens"):
            gamma = int(row["spec_tokens"])
            spec_by_batch_gamma[(batch, gamma)] = row

    decisions: list[dict[str, Any]] = []
    for batch in sorted(target_by_batch):
        target = target_by_batch[batch]
        target_tok_s = _steady_tok_s(target)
        best_action = "off"
        best_gamma = 0
        best_tok_s = target_tok_s
        candidates: dict[str, float] = {"off": target_tok_s}

        gammas = sorted(
            gamma
            for (candidate_batch, gamma) in spec_by_batch_gamma
            if candidate_batch == batch
        )
        for gamma in gammas:
            row = spec_by_batch_gamma[(batch, gamma)]
            tok_s = _steady_tok_s(row)
            candidates[f"gamma{gamma}"] = tok_s
            if tok_s > best_tok_s:
                best_action = f"gamma{gamma}"
                best_gamma = gamma
                best_tok_s = tok_s

        gain = best_tok_s / target_tok_s - 1.0
        if gain < min_gain:
            best_action = "off"
            best_gamma = 0
            best_tok_s = target_tok_s
            gain = 0.0

        decisions.append(
            {
                "batch": batch,
                "target_tok_s": target_tok_s,
                "best_action": best_action,
                "best_gamma": best_gamma,
                "best_tok_s": best_tok_s,
                "best_gain": gain,
                "candidates": candidates,
            }
        )
    return decisions


@dataclass
class GammaStats:
    steps: int = 0
    emitted_tokens: int = 0
    stage_time_ms: float = 0.0
    wall_steps: int = 0
    wall_emitted_tokens: int = 0
    wall_time_ms: float = 0.0
    accepted_draft_tokens: int = 0
    draft_positions_attempted: list[int] = field(default_factory=list)
    draft_positions_accepted: list[int] = field(default_factory=list)

    @property
    def tok_per_ms(self) -> float:
        if self.stage_time_ms <= 0:
            return 0.0
        return self.emitted_tokens / self.stage_time_ms

    @property
    def wall_tok_per_ms(self) -> float:
        if self.wall_time_ms <= 0:
            return 0.0
        return self.wall_emitted_tokens / self.wall_time_ms

    def ensure_gamma(self, gamma: int) -> None:
        while len(self.draft_positions_attempted) < gamma:
            self.draft_positions_attempted.append(0)
            self.draft_positions_accepted.append(0)

    def acceptance_by_position(self) -> list[float]:
        rates = []
        for accepted, attempted in zip(
            self.draft_positions_accepted, self.draft_positions_attempted
        ):
            rates.append(accepted / attempted if attempted else 0.0)
        return rates


class AdaptiveSpecGammaController:
    """Low-overhead hill-climbing controller over speculation actions."""

    def __init__(
        self,
        max_gamma: int,
        initial_gamma: int = 1,
        window_steps: int = 24,
        improvement_margin: float = 0.03,
        ewma_alpha: float = 0.35,
        allow_off: bool = False,
        request_level: bool = False,
    ) -> None:
        if max_gamma <= 0:
            raise ValueError("max_gamma must be positive")
        self.max_gamma = max_gamma
        self.allow_off = allow_off
        self.request_level = request_level
        min_gamma = 0 if allow_off else 1
        self.current_gamma = max(min_gamma, min(initial_gamma, max_gamma))
        self.window_steps = max(1, window_steps)
        self.improvement_margin = improvement_margin
        self.ewma_alpha = ewma_alpha
        self._window_seen = 0
        self.action_space = list(range(min_gamma, max_gamma + 1))
        self._probe_queue = [
            gamma for gamma in self.action_space
            if gamma != self.current_gamma
        ]
        if allow_off and 0 in self._probe_queue:
            # Probe real speculation windows before trying request-sticky off.
            self._probe_queue = [
                gamma for gamma in self._probe_queue if gamma != 0
            ] + [0]
        self.stats = {gamma: GammaStats() for gamma in self.action_space}
        self.ewma_reward = {gamma: 0.0 for gamma in self.action_space}
        self.wall_ewma_reward = {gamma: 0.0 for gamma in self.action_space}
        self.last_batch_size = 0
        self.last_selected_gamma = self.current_gamma

    def _has_observation(self, gamma: int) -> bool:
        stats = self.stats.get(gamma)
        return bool(
            stats is not None
            and (stats.steps > 0 or stats.wall_steps > 0)
        )

    def select_gamma(self, batch_size: int, scheduled_gamma: int) -> int:
        self.last_batch_size = batch_size
        min_gamma = 0 if self.allow_off else 1
        if (
            self.request_level
            and not self._probe_queue
            and self._has_observation(self.current_gamma)
        ):
            self._advance_policy()
        selected = max(min_gamma, min(self.current_gamma, scheduled_gamma,
                                      self.max_gamma))
        self.last_selected_gamma = selected
        if self.request_level and self._probe_queue:
            self.current_gamma = self._probe_queue.pop(0)
        return selected

    def record_step(self, gamma: int, accepted_token_ids: Any,
                    stage_times: tuple[float, float, float]) -> int:
        if gamma <= 0:
            return 0

        stats = self.stats[gamma]
        stats.ensure_gamma(gamma)

        token_rows = accepted_token_ids.detach().cpu().tolist()
        emitted = 0
        accepted_draft = 0
        for row in token_rows:
            non_padding = 0
            for token_id in row:
                if token_id == -1:
                    break
                non_padding += 1
            emitted += non_padding
            # Speculative decoding emits accepted draft tokens plus one
            # verifier token, either the replacement at first rejection or the
            # bonus token when all proposals are accepted.
            draft_count = min(max(non_padding - 1, 0), gamma)
            accepted_draft += draft_count
            for index in range(gamma):
                stats.draft_positions_attempted[index] += 1
                if draft_count > index:
                    stats.draft_positions_accepted[index] += 1

        proposal_avg_ms, scoring_ms, verification_ms = stage_times
        stage_time_ms = proposal_avg_ms * gamma + scoring_ms + verification_ms
        stats.steps += 1
        stats.emitted_tokens += emitted
        stats.accepted_draft_tokens += accepted_draft
        stats.stage_time_ms += max(stage_time_ms, 1e-9)

        reward = emitted / max(stage_time_ms, 1e-9)
        old_reward = self.ewma_reward[gamma]
        if old_reward == 0.0:
            self.ewma_reward[gamma] = reward
        else:
            self.ewma_reward[gamma] = (
                self.ewma_alpha * reward + (1.0 - self.ewma_alpha) * old_reward
            )

        if not self.request_level:
            self._window_seen += 1
            if self._window_seen >= self.window_steps:
                self._window_seen = 0
                self._advance_policy()

        return emitted

    def record_wall_step(self, gamma: int, emitted_tokens: int,
                         wall_time_ms: float) -> None:
        if gamma < 0 or emitted_tokens <= 0 or wall_time_ms <= 0:
            return
        if gamma not in self.stats:
            return

        stats = self.stats[gamma]
        stats.wall_steps += 1
        stats.wall_emitted_tokens += emitted_tokens
        stats.wall_time_ms += wall_time_ms

        reward = emitted_tokens / wall_time_ms
        old_reward = self.wall_ewma_reward[gamma]
        if old_reward == 0.0:
            self.wall_ewma_reward[gamma] = reward
        else:
            self.wall_ewma_reward[gamma] = (
                self.ewma_alpha * reward + (1.0 - self.ewma_alpha) * old_reward
            )

        if gamma == 0 and not self.request_level:
            self._window_seen += 1
            if self._window_seen >= self.window_steps:
                self._window_seen = 0
                self._advance_policy()

    def record_request(self, gamma: int, batch_size: int, emitted_tokens: int,
                       wall_time_ms: float) -> None:
        """Record one completed request batch for request-level policies."""
        if gamma < 0 or emitted_tokens <= 0 or wall_time_ms <= 0:
            return
        if gamma not in self.stats:
            return

        self.last_batch_size = batch_size
        stats = self.stats[gamma]
        stats.wall_steps += 1
        stats.wall_emitted_tokens += emitted_tokens
        stats.wall_time_ms += wall_time_ms

        reward = emitted_tokens / wall_time_ms
        old_reward = self.wall_ewma_reward[gamma]
        if old_reward == 0.0:
            self.wall_ewma_reward[gamma] = reward
        else:
            self.wall_ewma_reward[gamma] = (
                self.ewma_alpha * reward + (1.0 - self.ewma_alpha) * old_reward
            )

    def _advance_policy(self) -> None:
        reward_by_gamma = (
            self.wall_ewma_reward
            if any(reward > 0.0 for reward in self.wall_ewma_reward.values())
            else self.ewma_reward
        )
        observed = [
            gamma for gamma, reward in reward_by_gamma.items()
            if reward > 0.0
            and (self.stats[gamma].steps > 0 or self.stats[gamma].wall_steps > 0)
        ]
        if not observed:
            return

        # Probe each action once before settling. This is cheap because the
        # action set is tiny: off plus a few gamma values.
        if self._probe_queue:
            self.current_gamma = self._probe_queue.pop(0)
            return

        best_gamma = max(observed, key=lambda gamma: reward_by_gamma[gamma])
        best_reward = reward_by_gamma[best_gamma]
        acceptable_reward = best_reward / (1.0 + self.improvement_margin)
        # Prefer the cheapest action that is statistically close enough to the
        # fastest measured action. This avoids paying wider speculation for
        # small/noisy throughput deltas.
        for gamma in sorted(observed):
            if reward_by_gamma[gamma] >= acceptable_reward:
                self.current_gamma = gamma
                return
        self.current_gamma = best_gamma

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_gamma": self.current_gamma,
            "last_selected_gamma": self.last_selected_gamma,
            "last_batch_size": self.last_batch_size,
            "ewma_reward": self.ewma_reward,
            "wall_ewma_reward": self.wall_ewma_reward,
            "stats": {
                gamma: {
                    "steps": stats.steps,
                    "emitted_tokens": stats.emitted_tokens,
                    "accepted_draft_tokens": stats.accepted_draft_tokens,
                    "stage_time_ms": stats.stage_time_ms,
                    "tok_per_ms": stats.tok_per_ms,
                    "wall_steps": stats.wall_steps,
                    "wall_emitted_tokens": stats.wall_emitted_tokens,
                    "wall_time_ms": stats.wall_time_ms,
                    "wall_tok_per_ms": stats.wall_tok_per_ms,
                    "acceptance_by_position": stats.acceptance_by_position(),
                }
                for gamma, stats in self.stats.items()
            },
        }


class BatchAdaptiveSpecGammaController:
    """State-aware wrapper that learns one gamma controller per batch bin."""

    def __init__(
        self,
        max_gamma: int,
        initial_gamma: int = 1,
        window_steps: int = 24,
        improvement_margin: float = 0.03,
        ewma_alpha: float = 0.35,
        allow_off: bool = False,
        request_level: bool = False,
    ) -> None:
        self.max_gamma = max_gamma
        self.initial_gamma = initial_gamma
        self.window_steps = window_steps
        self.improvement_margin = improvement_margin
        self.ewma_alpha = ewma_alpha
        self.allow_off = allow_off
        self.request_level = request_level
        self.current_gamma = max(0 if allow_off else 1,
                                 min(initial_gamma, max_gamma))
        self.last_selected_gamma = self.current_gamma
        self._controllers: dict[str, AdaptiveSpecGammaController] = {}
        self._last_key = "uninitialized"

    @staticmethod
    def batch_key(batch_size: int) -> str:
        for upper in (1, 2, 4, 8, 16, 24, 32, 48, 64):
            if batch_size <= upper:
                return f"b<={upper}"
        return "b>64"

    def _controller_for(self, key: str) -> AdaptiveSpecGammaController:
        if key not in self._controllers:
            self._controllers[key] = AdaptiveSpecGammaController(
                max_gamma=self.max_gamma,
                initial_gamma=self.initial_gamma,
                window_steps=self.window_steps,
                improvement_margin=self.improvement_margin,
                ewma_alpha=self.ewma_alpha,
                allow_off=self.allow_off,
                request_level=self.request_level,
            )
        return self._controllers[key]

    def select_gamma(self, batch_size: int, scheduled_gamma: int) -> int:
        key = self.batch_key(batch_size)
        self._last_key = key
        controller = self._controller_for(key)
        self.current_gamma = controller.select_gamma(batch_size, scheduled_gamma)
        self.last_selected_gamma = controller.last_selected_gamma
        return self.current_gamma

    def record_step(self, gamma: int, accepted_token_ids: Any,
                    stage_times: tuple[float, float, float]) -> int:
        controller = self._controller_for(self._last_key)
        emitted = controller.record_step(gamma, accepted_token_ids, stage_times)
        self.current_gamma = controller.current_gamma
        self.last_selected_gamma = controller.last_selected_gamma
        return emitted

    def record_wall_step(self, gamma: int, emitted_tokens: int,
                         wall_time_ms: float) -> None:
        controller = self._controller_for(self._last_key)
        controller.record_wall_step(gamma, emitted_tokens, wall_time_ms)
        self.current_gamma = controller.current_gamma
        self.last_selected_gamma = controller.last_selected_gamma

    def record_request(self, gamma: int, batch_size: int, emitted_tokens: int,
                       wall_time_ms: float) -> None:
        key = self.batch_key(batch_size)
        self._last_key = key
        controller = self._controller_for(key)
        controller.record_request(gamma, batch_size, emitted_tokens, wall_time_ms)
        self.current_gamma = controller.current_gamma
        self.last_selected_gamma = controller.last_selected_gamma

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_gamma": self.current_gamma,
            "last_selected_gamma": self.last_selected_gamma,
            "last_state": self._last_key,
            "states": {
                key: controller.snapshot()
                for key, controller in sorted(self._controllers.items())
            },
        }


class AcceptanceTargetSpecGammaController:
    """Baseline that adapts gamma to hold draft acceptance near a target rate.

    This mirrors the policy shape of acceptance-targeting dynamic proposers
    (vLLM PR #26504, eagle_dynamic): widen the speculation window when the
    observed acceptance rate is above the target band, narrow it when below.
    It never consults throughput. It exists to test whether acceptance is a
    sufficient control signal, against the goodput-reward controller.
    """

    def __init__(
        self,
        max_gamma: int,
        initial_gamma: int = 1,
        target_acceptance: float = 0.7,
        band: float = 0.05,
        window_steps: int = 8,
        ewma_alpha: float = 0.35,
    ) -> None:
        if max_gamma <= 0:
            raise ValueError("max_gamma must be positive")
        self.max_gamma = max_gamma
        self.current_gamma = max(1, min(initial_gamma, max_gamma))
        self.target_acceptance = target_acceptance
        self.band = band
        self.window_steps = max(1, window_steps)
        self.ewma_alpha = ewma_alpha
        self.ewma_acceptance = 0.0
        self.request_level = False
        self._window_seen = 0
        self._observed_steps = 0
        self.last_batch_size = 0
        self.last_selected_gamma = self.current_gamma
        self.gamma_step_counts: dict[int, int] = {}

    def select_gamma(self, batch_size: int, scheduled_gamma: int) -> int:
        self.last_batch_size = batch_size
        selected = max(1, min(self.current_gamma, scheduled_gamma,
                              self.max_gamma))
        self.last_selected_gamma = selected
        return selected

    def record_step(self, gamma: int, accepted_token_ids: Any,
                    stage_times: tuple[float, float, float]) -> int:
        if gamma <= 0:
            return 0
        token_rows = accepted_token_ids.detach().cpu().tolist()
        emitted = 0
        accepted_draft = 0
        attempted_draft = 0
        for row in token_rows:
            non_padding = 0
            for token_id in row:
                if token_id == -1:
                    break
                non_padding += 1
            emitted += non_padding
            accepted_draft += min(max(non_padding - 1, 0), gamma)
            attempted_draft += gamma

        if attempted_draft:
            acceptance = accepted_draft / attempted_draft
            if self._observed_steps == 0:
                self.ewma_acceptance = acceptance
            else:
                self.ewma_acceptance = (
                    self.ewma_alpha * acceptance
                    + (1.0 - self.ewma_alpha) * self.ewma_acceptance
                )
            self._observed_steps += 1
            self.gamma_step_counts[gamma] = (
                self.gamma_step_counts.get(gamma, 0) + 1)

        self._window_seen += 1
        if self._window_seen >= self.window_steps:
            self._window_seen = 0
            self._advance_policy()
        return emitted

    def _advance_policy(self) -> None:
        if self._observed_steps == 0:
            return
        if (
            self.ewma_acceptance > self.target_acceptance + self.band
            and self.current_gamma < self.max_gamma
        ):
            self.current_gamma += 1
        elif (
            self.ewma_acceptance < self.target_acceptance - self.band
            and self.current_gamma > 1
        ):
            self.current_gamma -= 1

    def record_wall_step(self, gamma: int, emitted_tokens: int,
                         wall_time_ms: float) -> None:
        return None

    def record_request(self, gamma: int, batch_size: int, emitted_tokens: int,
                       wall_time_ms: float) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_gamma": self.current_gamma,
            "last_selected_gamma": self.last_selected_gamma,
            "last_batch_size": self.last_batch_size,
            "target_acceptance": self.target_acceptance,
            "band": self.band,
            "ewma_acceptance": self.ewma_acceptance,
            "observed_steps": self._observed_steps,
            "gamma_step_counts": self.gamma_step_counts,
        }


class FixedSpecGammaController:
    def __init__(self, gamma: int) -> None:
        if gamma < 0:
            raise ValueError("fixed runtime gamma must be non-negative")
        self.current_gamma = gamma

    def select_gamma(self, batch_size: int, scheduled_gamma: int) -> int:
        return max(0, min(self.current_gamma, scheduled_gamma))

    def record_step(self, gamma: int, accepted_token_ids: Any,
                    stage_times: tuple[float, float, float]) -> int:
        return 0

    def record_wall_step(self, gamma: int, emitted_tokens: int,
                         wall_time_ms: float) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {"current_gamma": self.current_gamma}


class BatchPolicySpecGammaController:
    """Static batch-size policy for hard-off smoke tests.

    The policy is a sorted list of ``(max_batch_size, gamma)`` pairs. The first
    pair whose max batch is >= the current active batch is selected.
    """

    def __init__(self, policy: list[tuple[int, int]]) -> None:
        if not policy:
            raise ValueError("batch policy must not be empty")
        for max_batch, gamma in policy:
            if max_batch <= 0:
                raise ValueError("batch policy thresholds must be positive")
            if gamma < 0:
                raise ValueError("batch policy gammas must be non-negative")
        self.policy = sorted(policy)
        self.current_gamma = 0
        self.last_batch_size = 0
        self.action_counts = {gamma: 0 for _, gamma in self.policy}

    @classmethod
    def parse(cls, text: str) -> "BatchPolicySpecGammaController":
        policy = []
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    "batch policy entries must look like '<max_batch>:<gamma>'")
            max_batch_text, gamma_text = item.split(":", 1)
            policy.append((int(max_batch_text), int(gamma_text)))
        return cls(policy)

    def select_gamma(self, batch_size: int, scheduled_gamma: int) -> int:
        self.last_batch_size = batch_size
        selected = self.policy[-1][1]
        for max_batch, gamma in self.policy:
            if batch_size <= max_batch:
                selected = gamma
                break
        self.current_gamma = max(0, min(selected, scheduled_gamma))
        self.action_counts[self.current_gamma] = (
            self.action_counts.get(self.current_gamma, 0) + 1)
        return self.current_gamma

    def record_step(self, gamma: int, accepted_token_ids: Any,
                    stage_times: tuple[float, float, float]) -> int:
        return 0

    def record_wall_step(self, gamma: int, emitted_tokens: int,
                         wall_time_ms: float) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_gamma": self.current_gamma,
            "last_batch_size": self.last_batch_size,
            "policy": self.policy,
            "action_counts": self.action_counts,
        }


def _count_sampler_output_tokens(outputs: Any) -> int:
    if not outputs:
        return 0
    emitted = 0
    for sampler_output in outputs:
        for group_output in getattr(sampler_output, "outputs", []) or []:
            samples = getattr(group_output, "samples", None)
            if samples is not None:
                emitted += len(samples)
            else:
                emitted += 1
    return emitted


def install_vllm_spec_gamma_controller(
    llm: Any,
    controller: Any,
    disable_vllm_spec_metrics: bool = False,
    hard_off: bool = False,
) -> Any:
    """Install a runtime gamma controller on a vLLM 0.6.x LLM instance."""
    worker = getattr(llm.llm_engine.model_executor, "driver_worker", None)
    if worker is None:
        raise RuntimeError("could not find vLLM driver_worker")
    if not hasattr(worker, "_run_speculative_decoding_step"):
        raise RuntimeError("driver_worker does not look like a SpecDecodeWorker")

    if disable_vllm_spec_metrics and hasattr(worker, "_metrics"):
        worker._metrics.maybe_collect_rejsample_metrics = lambda k: None

    original_execute_model = worker.execute_model
    original_create_outputs = worker._create_output_sampler_list
    original_run_no_spec = worker._run_no_spec
    request_level_controller = bool(getattr(controller, "request_level", False))

    def controlled_run_no_spec(self: Any, execute_model_req: Any,
                               skip_proposer: bool) -> Any:
        if hard_off and getattr(self, "_auto_spec_force_skip_proposer", False):
            skip_proposer = True
        return original_run_no_spec(execute_model_req, skip_proposer)

    def controlled_execute_model(self: Any, execute_model_req: Any = None) -> Any:
        speculative_step = (
            execute_model_req is not None
            and execute_model_req.num_lookahead_slots > 0
        )
        selected_gamma = None
        request_ids = []
        if execute_model_req is not None and execute_model_req.num_lookahead_slots > 0:
            batch_size = len(execute_model_req.seq_group_metadata_list or [])
            request_ids = [
                seq_group_metadata.request_id
                for seq_group_metadata in execute_model_req.seq_group_metadata_list
            ]
            scheduled_gamma = int(execute_model_req.num_lookahead_slots)
            if hard_off:
                request_actions = getattr(
                    self, "_auto_spec_request_actions", {})
                new_request_ids = [
                    request_id for request_id in request_ids
                    if request_id not in request_actions
                ]
                if new_request_ids:
                    gamma = int(
                        controller.select_gamma(batch_size, scheduled_gamma))
                    for request_id in new_request_ids:
                        request_actions[request_id] = gamma
                    self._auto_spec_request_actions = request_actions
                assigned_gammas = [
                    int(request_actions.get(request_id, scheduled_gamma))
                    for request_id in request_ids
                ]
                selected_gamma = max(assigned_gammas) if assigned_gammas else 0
                execute_model_req.num_lookahead_slots = selected_gamma
                for seq_group_metadata, gamma in zip(
                        execute_model_req.seq_group_metadata_list,
                        assigned_gammas):
                    if seq_group_metadata.num_speculative_tokens is not None:
                        seq_group_metadata.num_speculative_tokens = gamma
            else:
                gamma = int(controller.select_gamma(batch_size, scheduled_gamma))
                selected_gamma = gamma
                execute_model_req.num_lookahead_slots = gamma
                for seq_group_metadata in execute_model_req.seq_group_metadata_list:
                    if seq_group_metadata.num_speculative_tokens is not None:
                        seq_group_metadata.num_speculative_tokens = gamma
        start = time.perf_counter()
        self._auto_spec_pending_step = None
        self._auto_spec_force_skip_proposer = bool(
            hard_off and selected_gamma == 0)
        self._auto_spec_last_selected_gamma = selected_gamma
        outputs = None
        try:
            outputs = original_execute_model(execute_model_req)
            return outputs
        finally:
            if speculative_step:
                if not request_level_controller:
                    pending_step = getattr(self, "_auto_spec_pending_step", None)
                    if pending_step is not None and hasattr(controller,
                                                            "record_wall_step"):
                        wall_time_ms = (time.perf_counter() - start) * 1000.0
                        controller.record_wall_step(
                            pending_step["gamma"],
                            pending_step["emitted_tokens"],
                            wall_time_ms,
                        )
                    elif selected_gamma == 0 and hasattr(controller,
                                                         "record_wall_step"):
                        wall_time_ms = (time.perf_counter() - start) * 1000.0
                        controller.record_wall_step(
                            0,
                            _count_sampler_output_tokens(outputs),
                            wall_time_ms,
                        )
                self._auto_spec_pending_step = None
                self._auto_spec_force_skip_proposer = False

    def controlled_create_outputs(
        self: Any,
        seq_group_metadata_list: Any,
        accepted_token_ids: Any,
        target_logprobs: Any,
        k: int,
        stage_times: tuple[float, float, float],
    ) -> Any:
        emitted_tokens = controller.record_step(
            int(k), accepted_token_ids, stage_times)
        self._auto_spec_pending_step = {
            "gamma": int(k),
            "emitted_tokens": emitted_tokens,
        }
        return original_create_outputs(
            seq_group_metadata_list,
            accepted_token_ids,
            target_logprobs,
            k,
            stage_times,
        )

    worker.execute_model = MethodType(controlled_execute_model, worker)
    worker._run_no_spec = MethodType(controlled_run_no_spec, worker)
    worker._create_output_sampler_list = MethodType(controlled_create_outputs, worker)
    worker._auto_spec_gamma_controller = controller
    return controller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--min-gain", type=float, default=0.03)
    args = parser.parse_args()

    decisions = summarize_best_actions(args.logs, min_gain=args.min_gain)
    print(json.dumps(decisions, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
