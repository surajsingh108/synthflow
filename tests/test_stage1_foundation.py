"""
Stage 1 tests – Foundation

Tests that the package installs, imports cleanly, exposes the right
public API, and that exceptions behave correctly.

DO NOT MODIFY THIS FILE.
Fix the implementation, not the tests.
"""

import pytest


# — 1. Import tests —————————————————————————————————————————————————————————

class TestImports:
    def test_import_synthflow(self):
        import synthflow
        assert synthflow is not None

    def test_version_exists(self):
        import synthflow
        assert hasattr(synthflow, "__version__")
        assert isinstance(synthflow.__version__, str)
        assert synthflow.__version__ == "0.1.0"

    def test_synflow_importable_from_top(self):
        from synthflow import SynFlow
        assert SynFlow is not None

    def test_all_exceptions_importable(self):
        from synthflow import (
            SynError,
            SynConfigError,
            SynIngestError,
            SynImpError,
            SynBackendError,
            SynRouterError,
        )
        for exc in [
            SynError, SynConfigError, SynIngestError,
            SynImpError, SynBackendError, SynRouterError,
        ]:
            assert exc is not None

    def test_dunder_all_complete(self):
        import synthflow
        assert "SynFlow" in synthflow.__all__
        assert "SynError" in synthflow.__all__
        assert "SynConfigError" in synthflow.__all__


# — 2. Exception hierarchy tests ————————————————————————————————————————————

class TestExceptionHierarchy:
    def test_base_exception_is_exception(self):
        from synthflow import SynError
        assert issubclass(SynError, Exception)

    def test_all_exceptions_inherit_from_syn_error(self):
        from synthflow import (
            SynConfigError, SynIngestError, SynImpError,
            SynBackendError, SynRouterError,
        )
        from synthflow.exceptions import SynError
        for exc_class in [
            SynConfigError, SynIngestError, SynImpError,
            SynBackendError, SynRouterError,
        ]:
            assert issubclass(exc_class, SynError), (
                f"{exc_class.__name__} must inherit from SynError"
            )

    def test_exception_message(self):
        from synthflow import SynError
        err = SynError("something went wrong")
        assert str(err) == "something went wrong"

    def test_exception_message_with_detail(self):
        from synthflow import SynConfigError
        err = SynConfigError("invalid domain", detail="must be industrial or iot")
        assert "invalid domain" in str(err)
        assert "must be industrial or iot" in str(err)

    def test_exceptions_are_catchable_as_base(self):
        from synthflow import SynError, SynConfigError
        with pytest.raises(SynError):
            raise SynConfigError("test")

    def test_each_exception_has_message_attr(self):
        from synthflow import SynIngestError
        err = SynIngestError("file not found")
        assert hasattr(err, "message")
        assert err.message == "file not found"

    def test_each_exception_has_detail_attr(self):
        from synthflow import SynBackendError
        err = SynBackendError("OOM", detail="reduce batch size")
        assert hasattr(err, "detail")
        assert err.detail == "reduce batch size"

    def test_detail_defaults_to_empty_string(self):
        from synthflow import SynRouterError
        err = SynRouterError("no backend found")
        assert err.detail == ""


# — 3. SynFlow instantiation tests ——————————————————————————————————————————

class TestSynFlowInstantiation:
    def test_manual_mode_with_stub_config(self):
        from synthflow import SynFlow
        # config just needs to be truthy for Stage 1
        sf = SynFlow(mode="manual", config={"domain": "industrial"})
        assert sf is not None

    def test_manual_mode_stores_mode(self):
        from synthflow import SynFlow
        sf = SynFlow(mode="manual", config={"stub": True})
        assert sf.mode == "manual"

    def test_manual_mode_stores_config(self):
        from synthflow import SynFlow
        cfg = {"domain": "industrial"}
        sf = SynFlow(mode="manual", config=cfg)
        assert sf.config == cfg

    def test_manual_mode_stores_data_path(self):
        from synthflow import SynFlow
        sf = SynFlow(mode="manual", config={"x": 1}, data="my_data.csv")
        assert sf.data == "my_data.csv"

    def test_manual_mode_no_config_raises(self):
        from synthflow import SynFlow, SynConfigError
        with pytest.raises(SynConfigError):
            SynFlow(mode="manual")

    def test_auto_mode_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from synthflow import SynFlow, SynConfigError
        with pytest.raises(SynConfigError):
            SynFlow(mode="auto")

    def test_auto_mode_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        from synthflow import SynFlow
        sf = SynFlow(mode="auto")
        assert sf.mode == "auto"
        assert sf.api_key == "sk-ant-test-key"

    def test_auto_mode_explicit_key(self):
        from synthflow import SynFlow
        sf = SynFlow(mode="auto", api_key="sk-ant-explicit")
        assert sf.api_key == "sk-ant-explicit"

    def test_invalid_mode_raises(self):
        from synthflow import SynFlow, SynConfigError
        with pytest.raises(SynConfigError):
            SynFlow(mode="invalid")

    def test_repr_contains_mode(self):
        from synthflow import SynFlow
        sf = SynFlow(mode="manual", config={"x": 1})
        assert "manual" in repr(sf)

    def test_version_constant(self):
        from synthflow import SynFlow
        assert SynFlow.VERSION == "0.1.0"

    def test_chat_raises_not_implemented(self):
        from synthflow import SynFlow, SynConfigError
        sf = SynFlow(mode="manual", config={"x": 1})
        with pytest.raises(SynConfigError):
            sf.chat("hello")

    def test_generate_raises_not_implemented(self):
        from synthflow import SynFlow
        sf = SynFlow(mode="manual", config={"x": 1})
        with pytest.raises(NotImplementedError):
            sf.generate()
