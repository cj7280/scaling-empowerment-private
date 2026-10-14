"""Every SkillLearner construction in src/ passes the engine's ema_tau_one_shortcut."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "empowerment_warp"

# (file, enclosing function, ema_tau_one_shortcut). SAC always runs the Polyak
# expression; the PQN engine copies the live encoder outright at target_tau == 1.0.
EXPECTED = {
    ("agents/see_sac.py", "create_learners", False),
    ("training/pqn/runner.py", "run_potential_pqn_training", True),
}


def _construction_sites():
    sites = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
                if name != "SkillLearner":
                    continue
                flags = [k for k in node.keywords if k.arg == "ema_tau_one_shortcut"]
                assert len(flags) == 1 and isinstance(flags[0].value, ast.Constant), (
                    f"{path}:{node.lineno} must pass ema_tau_one_shortcut as a literal")
                sites.append((path.relative_to(SRC).as_posix(), function.name, flags[0].value.value, node.lineno))
    return sites


def test_skill_learner_construction_sites_pass_engine_flag() -> None:
    sites = _construction_sites()
    # A call inside a nested function is seen from both functions; keep the innermost.
    innermost = {}
    for file, function, flag, line in sites:
        innermost[(file, line)] = (file, function, flag)
    assert set(innermost.values()) == EXPECTED
    assert len(innermost) == len(EXPECTED)
