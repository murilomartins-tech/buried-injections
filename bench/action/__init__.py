"""Action-level metric: score the tool call an agent makes, not just the text it read.

    canon.py      declared, versioned canonicalizer (canon-v1)
    evaluator.py  deterministic pass/fail + attribution over a trace (no LLM)
    simulate.py   AgentDojo tools + scripted agent -> traces (no LLM)

See bench/action/README.md and issue #9.
"""
