"""
tests/conftest.py

Shared pytest configuration.

Custom markers
--------------
  e2e   — full end-to-end tests that load real models / start servers.
  slow  — tests that run for more than a few seconds.

Register them so ``pytest --strict-markers`` never complains:
  pytest -m "not e2e and not slow"   → fast unit-test suite only
  pytest -m e2e                      → smoke + integration tests only

Installs lightweight stub modules for heavy optional dependencies
(cv2, openai, anthropic, etc.) so that any test file can safely
import modules that reference these at import time, regardless of
collection order.

Using importlib.util.spec_from_loader ensures the stubs have a valid
__spec__ and do not trigger pytest's ".__spec__ is None" guard.
"""

import sys
import types
import importlib.util
from unittest.mock import MagicMock


def pytest_configure(config):
    """Register custom markers so --strict-markers never raises."""
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end test that loads real models or starts external processes "
        "(excluded from the default fast-test suite).",
    )
    config.addinivalue_line(
        "markers",
        "slow: test that takes more than a few seconds to run.",
    )


def _make_stub(name: str) -> types.ModuleType:
    """Create a module stub with a valid __spec__ so pytest won't complain."""
    mod = types.ModuleType(name)
    # Give it a minimal but valid spec
    loader = importlib.util.find_spec("types")  # borrow any real loader
    spec = importlib.machinery.ModuleSpec(name, loader=None)
    mod.__spec__ = spec
    mod.__path__ = []       # make it look like a package too
    return mod


_HEAVY_DEPS = [
    "google",
    "google.generativeai",
    "openai",
    "anthropic",
    "lmdeploy",
    "pydantic",
    "typing_extensions",
    "cv2",
]

for _dep in _HEAVY_DEPS:
    if _dep not in sys.modules:
        sys.modules[_dep] = _make_stub(_dep)

# Extra attributes needed by planner_utils at import time
_openai = sys.modules["openai"]
if not hasattr(_openai, "OpenAI"):
    _openai.OpenAI = object
if not hasattr(_openai, "AzureOpenAI"):
    _openai.AzureOpenAI = object

_pydantic = sys.modules["pydantic"]
if not hasattr(_pydantic, "BaseModel"):
    _pydantic.BaseModel = object
if not hasattr(_pydantic, "Field"):
    _pydantic.Field = lambda *a, **kw: None

_typing_ext = sys.modules.get("typing_extensions")
if _typing_ext is None:
    sys.modules["typing_extensions"] = _make_stub("typing_extensions")
    _typing_ext = sys.modules["typing_extensions"]
if not hasattr(_typing_ext, "TypedDict"):
    import typing
    _typing_ext.TypedDict = typing.TypedDict
if not hasattr(_typing_ext, "Required"):
    _typing_ext.Required = lambda t: t
if not hasattr(_typing_ext, "Annotated"):
    import typing
    _typing_ext.Annotated = getattr(typing, "Annotated", lambda t, *_: t)

_lmdeploy = sys.modules["lmdeploy"]
if not hasattr(_lmdeploy, "pipeline"):
    _lmdeploy.pipeline = lambda *a, **kw: None
if not hasattr(_lmdeploy, "TurbomindEngineConfig"):
    _lmdeploy.TurbomindEngineConfig = object
if not hasattr(_lmdeploy, "VisionConfig"):
    _lmdeploy.VisionConfig = object
if not hasattr(_lmdeploy, "GenerationConfig"):
    _lmdeploy.GenerationConfig = object
if not hasattr(_lmdeploy, "PytorchEngineConfig"):
    _lmdeploy.PytorchEngineConfig = object

# ---------------------------------------------------------------------------
# Pre-load the real planner_utils so that test-file stubs using setdefault()
# cannot replace it with a minimal stub.  This prevents cross-test import
# failures when tests that stub planner_utils run before tests that need the
# full symbol set (convert_format_2claude, ActionPlan_*, etc.).
# ---------------------------------------------------------------------------
if "embodiedbench.planner.planner_utils" not in sys.modules:
    try:
        import embodiedbench.planner.planner_utils  # noqa: F401
    except Exception:
        pass  # best-effort; individual test files may still stub it if needed
