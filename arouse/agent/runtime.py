"""Agent runtime: model turn -> validated action -> tool execution -> observation -> repeat.

Defence in depth against unreliable output:
  1. token mask: the model cannot emit input-only tokens (no forged tool results)
  2. protocol validation: every turn must parse into exactly one schema-valid action,
     tool calls must match the tool's argument schema; otherwise resample (bounded)
  3. copy-constrained decoding + grounding: free-text arguments (reminder task, note
     text, file path) can only be copied from the conversation; final answers may only
     use Arouse's response vocabulary plus words from the conversation / tool results;
     anything ungrounded is resampled (greedy result kept if no sample is grounded)
  4. completion guard: `finish` right after a failed tool call (with no success since)
     is converted to `fail`, so a false "Done." never reaches the user
  5. step limit: at most `max_tool_calls` tool calls per user message
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from arouse.agent.constraints import CopyConstraint
from arouse.agent.episode import Header, encode_prompt, turn_event
from arouse.agent.grounding import answer_issue, grounding_issue
from arouse.inference import InferenceEngine, SamplingParams
from arouse.protocol import Action, ProtocolError, ToolRegistry, Turn, decode_turn
from arouse.tokenizer import Special

Executor = Callable[[str, dict[str, Any]], tuple[bool, dict[str, Any]]]


@dataclasses.dataclass
class TurnResult:
    turn: Turn
    attempts: int
    raw_text: str
    valid: bool  # False = model never produced a valid turn (runtime substituted a fail)
    guarded: bool = False  # completion guard rewrote a false finish
    grounded: bool = True  # False = kept an ungrounded tool call (no grounded alternative found)


@dataclasses.dataclass
class RunResult:
    events: list[dict[str, Any]]  # new events appended during this run
    final: Action
    turns: list[TurnResult]


def last_observation(events: list[dict[str, Any]]) -> str | None:
    """Type of the latest tool observation since the last user message."""
    for ev in reversed(events):
        if ev["type"] == "user":
            return None
        if ev["type"] in ("tool_result", "tool_error"):
            return ev["type"]
    return None


class AgentRuntime:
    def __init__(
        self,
        engine: InferenceEngine,
        registry: ToolRegistry,
        *,
        max_tool_calls: int = 6,
        max_new_tokens: int = 200,
        retries: int = 2,
        guard_completion: bool = True,
        grounding_samples: int = 4,
        constrain_copy: bool = True,
    ) -> None:
        self.engine = engine
        self.registry = registry
        self.max_tool_calls = max_tool_calls
        self.max_new_tokens = max_new_tokens
        self.retries = retries
        self.guard_completion = guard_completion
        self.grounding_samples = grounding_samples
        self.constrain_copy = constrain_copy

    MIN_RESERVE = 48  # generation room kept even when older exchanges cannot be dropped

    def _fit(self, header: Header, events: list[dict[str, Any]]) -> list[int]:
        """Prompt ids; drops the oldest whole user exchanges if the context is too long."""
        tok = self.engine.tokenizer
        budget = self.engine.context_length - self.max_new_tokens
        evs = list(events)
        while True:
            ids = encode_prompt(tok, header, evs)
            if len(ids) <= budget:
                return ids
            nxt = next((i for i, e in enumerate(evs) if i > 0 and e["type"] == "user"), None)
            if nxt is None:
                if len(ids) <= self.engine.context_length - self.MIN_RESERVE:
                    return ids  # current request only: use the remaining context for the reply
                raise ProtocolError(f"conversation needs {len(ids)} tokens; the model allows {budget}")
            evs = evs[nxt:]

    def _sample(self, prompt: list[int], attempt: int, hook: CopyConstraint | None) -> tuple[Turn | None, str]:
        params = SamplingParams(max_new_tokens=self.max_new_tokens, temperature=0.0 if attempt == 0 else 0.7,
                                top_p=0.95, seed=attempt)
        gen = self.engine.generate_ids(prompt, params, stop_ids=[Special.END], banned_ids=[Special.EOS], logits_hook=hook)
        try:
            turn = decode_turn(self.engine.tokenizer, gen.token_ids)
            if turn.action.type == "tool_call":
                self.registry.validate_call(turn.action.tool, turn.action.arguments)
        except ProtocolError:
            return None, gen.text
        return turn, gen.text

    def next_turn(self, header: Header, events: list[dict[str, Any]]) -> TurnResult:
        prompt = self._fit(header, events)
        hook = CopyConstraint(self.engine.tokenizer, events, header.context) if self.constrain_copy else None
        raw = ""
        first_valid: TurnResult | None = None
        attempt = 0
        # Valid-output retries, then extra samples only while looking for a grounded tool call.
        while attempt < 1 + self.retries + (self.grounding_samples if first_valid else 0):
            turn, raw = self._sample(prompt, attempt, hook)
            attempt += 1
            if turn is None:
                continue
            result = TurnResult(turn, attempt, raw, True)
            if self.retries == 0 or (grounding_issue(turn.action, events) or answer_issue(turn.action, events)) is None:
                return self._guard(result, events)
            if first_valid is None:
                first_valid = dataclasses.replace(result, grounded=False)
        if first_valid is not None:
            return self._guard(dataclasses.replace(first_valid, attempts=attempt), events)
        fallback = Turn(Action.fail("Sorry, I couldn't work out a valid next step for that request."),
                        verify="runtime: the model did not produce a valid action")
        return TurnResult(fallback, attempt, raw, False)

    def _guard(self, r: TurnResult, events: list[dict[str, Any]]) -> TurnResult:
        if self.guard_completion and r.turn.action.type == "finish" and last_observation(events) == "tool_error":
            fixed = Turn(Action.fail("The last step failed, so the task was not completed."),
                         plan=r.turn.plan, verify="runtime guard: finish after a failed tool call")
            return dataclasses.replace(r, turn=fixed, guarded=True)
        return r

    def run(self, header: Header, events: list[dict[str, Any]], execute: Executor) -> RunResult:
        """Continue the conversation (whose last event is usually a user message) until a
        terminal action (ask_user / finish / fail)."""
        history = list(events)
        new: list[dict[str, Any]] = []
        turns: list[TurnResult] = []
        calls = 0
        while True:
            try:
                r = self.next_turn(header, history)
            except ProtocolError:
                if not new and not turns:
                    raise  # nothing happened yet: the request itself does not fit (caller decides)
                r = TurnResult(Turn(Action.fail("This conversation is too long for me to continue. Please start a new chat."),
                                    verify="runtime: context full"), 0, "", True)
            turns.append(r)
            ev = turn_event(r.turn)
            history.append(ev)
            new.append(ev)
            action = r.turn.action
            if action.is_terminal:
                return RunResult(new, action, turns)
            if calls >= self.max_tool_calls:
                stop = Turn(Action.fail("I stopped because the task needed too many steps."), verify="runtime: step limit")
                ev = turn_event(stop)
                history.append(ev)
                new.append(ev)
                return RunResult(new, stop.action, turns)
            calls += 1
            ok, payload = execute(action.tool, action.arguments)
            obs = {"type": "tool_result" if ok else "tool_error", "content": payload}
            history.append(obs)
            new.append(obs)
