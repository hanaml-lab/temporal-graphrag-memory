import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reconcile_assertions_on_the_bundled_account():
    path = ROOT / "examples" / "data" / "account_assertions.jsonl"
    out = _example("reconcile_assertions").main([str(path), "--as-of", "100"])
    plan = out[("acme", "plan")]
    # the late disclosure dates the upgrade to pro back to day 60, and the plan to move is not taken as a move
    assert plan.value_at(70) == "pro" and plan.value_at(200) == "pro" and plan.value_at(250) == "enterprise"
    assert plan.value_at(10) == "starter"
    assert out[("acme", "billing_city")].value_at(250) == "Denver"
