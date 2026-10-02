"""
simulate.py -- deterministic simulated tool layer over AgentDojo, producing traces.

No LLM. The tools are AgentDojo's real banking functions run against its real
environment, so tool outputs are exactly what an agent would see. The *agent* is
scripted from AgentDojo's own ground truth:

  user_only     executes the user task's ground-truth calls. An agent that ignores
                the injection.
  compromised   executes the user task's ground-truth calls, and right after the
                tool output that carries the injection, executes the injection
                task's ground-truth calls, then resumes the user task. An agent
                that fully complies (worst case; real compliance needs live mode).

Gateway: a text-level detector in front of tool outputs. If it flags an output,
the output is quarantined (the agent never sees it) and the scripted agent stops,
since the rest of its plan depended on that output. That stop is what makes a
false alarm visible at the action level: the user's task does not get done.

The injection is appended to the vector's normal content (buried, like the
leaderboard), not substituted for it, so the bill still carries the real IBAN.

Each step records `origin` (user_plan / injected_plan). The evaluator never reads
it; it is ground truth for checking attribution.
"""

import os

import yaml

from . import canon

USER_NAME, MODEL_NAME = "Emma Johnson", "the AI assistant"   # same as the leaderboard
SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "scenarios")


def _flat(text):
    """Undo YAML wrapping / escaping so the injected goal can be found in a tool output."""
    return canon.generic(text.replace("''", "'"))


FINANCIAL_WRITES = ("send_money", "schedule_transaction", "update_scheduled_transaction")


def _amount_error(fc):
    """A planning error: the right financial call with the amount off by one."""
    if fc.function in FINANCIAL_WRITES and "amount" in fc.args:
        args = dict(fc.args)
        args["amount"] = float(args["amount"]) + 1
        return fc.model_copy(update={"args": args})
    return fc


def _iban_typo(fc):
    """A planning error: the right financial call with the payee's last IBAN digit garbled
    (one edit away from the legitimate IBAN, which may sit in an untrusted tool output)."""
    r = fc.args.get("recipient") if fc.function in FINANCIAL_WRITES else None
    if isinstance(r, str) and r[-1:].isdigit():
        args = dict(fc.args)
        args["recipient"] = r[:-1] + str((int(r[-1]) + 1) % 10)
        return fc.model_copy(update={"args": args})
    return fc


PLANNING_ERRORS = {"user_error": _amount_error, "user_error_iban": _iban_typo}


def load_scenario(path):
    with open(path) as fh:
        scenario = yaml.safe_load(fh)
    with open(os.path.join(os.path.dirname(path), scenario["policy"])) as fh:
        policy = yaml.safe_load(fh)
    return scenario, policy


def scenario_paths(suite="banking"):
    d = os.path.join(SCENARIO_DIR, suite)
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".yaml") and not f.startswith("_")]


def _suite(name):
    from agentdojo.task_suite.load_suites import get_suites
    from datasets import AGENTDOJO_VERSION   # bench/ is on sys.path, as for run.py
    return get_suites(AGENTDOJO_VERSION)[name]


def payload_for(injection_task):
    from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
    return ImportantInstructionsAttack._JB_STRING.format(
        user=USER_NAME, model=MODEL_NAME, goal=injection_task.GOAL)


def build_trace(suite_name, user_task_id, injection_task_id=None, agent="compromised",
                block_injected_output=False, block_first_output=False, goal_rewrite=None):
    """Run one scripted episode and return its trace.

    block_injected_output  gateway flags the output carrying the injection (a catch)
    block_first_output     gateway flags the first tool output (a false alarm on benign traffic)
    goal_rewrite           optional fn(goal_text) -> goal_text: mutates how the attacker WRITES
                           the goal (e.g. a spaced or homoglyph IBAN). The scripted agent still
                           emits the canonical value, as a model that normalises it would.
    agent                  user_only | compromised | user_error | user_error_iban
                           The two user_error agents follow the user plan but garble the
                           financial call (amount + 1, or a one-digit IBAN typo): planning
                           errors, not injections -- the controls for over-attribution.
    """
    from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
    from agentdojo.functions_runtime import FunctionsRuntime

    suite = _suite(suite_name)
    user_task = suite.user_tasks[user_task_id]
    injection_task = suite.injection_tasks[injection_task_id] if injection_task_id else None

    injections = {}
    marker = None
    if injection_task:
        goal = injection_task.GOAL if goal_rewrite is None else goal_rewrite(injection_task.GOAL)
        payload = payload_for(injection_task).replace(injection_task.GOAL, goal)
        marker = goal
        injections = {v: d + payload for v, d in suite.get_injection_vector_defaults().items()}
    env = suite.load_and_inject_default_environment(injections)
    runtime = FunctionsRuntime(suite.tools)

    spans = [{"id": "s0", "source": "user", "trust": "trusted", "text": user_task.PROMPT}]
    steps, visible = [], ["s0"]
    plan = [("user_plan", c) for c in user_task.ground_truth(env.model_copy(deep=True))]
    if agent in PLANNING_ERRORS:
        err = PLANNING_ERRORS[agent]
        plan = [(o, c) if err(c) is c else ("planning_error", err(c)) for o, c in plan]
    injected_seen = False
    halted = None

    i = 0
    while i < len(plan):
        origin, fc = plan[i]
        call = {"tool": fc.function, "args": {k: v for k, v in dict(fc.args).items()}}
        result, error = runtime.run_function(env, fc.function, dict(fc.args))
        text = error if error else tool_result_to_str(result)
        span_id = f"s{len(spans)}"
        carries_injection = bool(marker) and _flat(marker) in _flat(text)
        blocked = (carries_injection and block_injected_output) or (block_first_output and not steps)

        spans.append({"id": span_id, "source": f"tool:{fc.function}",
                      "trust": "quarantined" if blocked else "untrusted",
                      "text": "" if blocked else text, "carries_injection": carries_injection})
        steps.append({"step": len(steps) + 1, "context_span_ids": list(visible), "call": call,
                      "origin": origin, "output_span_id": span_id, "output_blocked": blocked})
        if blocked:
            halted = "gateway_blocked_output"
            break
        visible.append(span_id)

        if carries_injection and not injected_seen:
            injected_seen = True
            if agent == "compromised":
                hijack = [("injected_plan", c) for c in injection_task.ground_truth(env.model_copy(deep=True))]
                plan[i + 1:i + 1] = hijack
        i += 1

    return {
        "trace_id": f"{suite_name}/{user_task_id}/{injection_task_id or 'none'}/{agent}"
                    f"{'/blocked' if block_injected_output or block_first_output else ''}",
        "suite": suite_name, "user_task": user_task_id, "injection_task": injection_task_id,
        "agent": f"simulated:{agent}",
        "injection_reached_agent": injected_seen,
        "halted": halted,
        "spans": spans,
        "steps": steps,
    }
