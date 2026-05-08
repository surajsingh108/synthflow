"""
Stage 9 tests – Core + Chat

Tests:
  - Magic word detection (all four words)
  - SynSession add, clear, len, as_messages
  - SynState transitions, reset, config setter
  - Override detection (model, sampling_rate, n_samples)
  - JSON repair in guardrails
  - SynParser.parse() with mocked Claude API
  - SynFlow manual mode: generate() returns SynResult
  - SynFlow auto mode: chat() state transitions (mocked API)
  - Magic word "generate" triggers pipeline (mocked)
  - Magic word "reset" wipes state
  - Magic word "show config" returns summary
  - Override applied mid-conversation

DO NOT MODIFY THIS FILE.
Fix the implementation, not the tests.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# – fixtures –

def make_csv_file(tmp_path, n=200):
    t = np.linspace(0, 4*np.pi, n)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="10ms"),
        "accel_x": np.sin(t) + 0.05*np.random.randn(n),
        "accel_y": np.cos(t) + 0.05*np.random.randn(n),
        "temperature": np.random.uniform(20, 80, n),
    })
    path = tmp_path / "sensor.csv"
    df.to_csv(path, index=False)
    return path


def mock_anthropic_response(patch_dict: dict):
    """Build a mock anthropic response returning the given patch dict."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = json.dumps(patch_dict)
    return mock_response


# – 1. Import –

class TestImport:
    def test_syn_flow_importable(self):
        from synthflow import SynFlow
        assert SynFlow is not None

    def test_magic_words_importable(self):
        from synthflow.chat.magic_words import detect_magic_word
        assert detect_magic_word is not None

    def test_syn_session_importable(self):
        from synthflow.chat.session import SynSession
        assert SynSession is not None

    def test_syn_state_importable(self):
        from synthflow.chat.state_machine import SynState
        assert SynState is not None

    def test_overrides_importable(self):
        from synthflow.chat.overrides import detect_override
        assert detect_override is not None

    def test_syn_parser_importable(self):
        from synthflow.parser import SynParser
        assert SynParser is not None


# – 2. Magic words –

class TestMagicWords:
    def test_generate_detected(self):
        from synthflow.chat.magic_words import detect_magic_word
        assert detect_magic_word("generate") == "generate"

    def test_generate_case_insensitive(self):
        from synthflow.chat.magic_words import detect_magic_word
        assert detect_magic_word("GENERATE") == "generate"

    def test_go_triggers_generate(self):
        from synthflow.chat.magic_words import detect_magic_word
        assert detect_magic_word("go") == "generate"

    def test_reset_detected(self):
        from synthflow.chat.magic_words import detect_magic_word
        assert detect_magic_word("reset") == "reset"

    def test_start_over_triggers_reset(self):
        from synthflow.chat.magic_words import detect_magic_word
        assert detect_magic_word("start over") == "reset"

    def test_show_config_detected(self):
        from synthflow.chat.magic_words import detect_magic_word
        assert detect_magic_word("show config") == "show config"

    def test_explain_detected(self):
        from synthflow.chat.magic_words import detect_magic_word
        assert detect_magic_word("explain") == "explain"

    def test_normal_message_returns_none(self):
        from synthflow.chat.magic_words import detect_magic_word
        assert detect_magic_word("accelerometer from a wind turbine") is None

    def test_empty_message_returns_none(self):
        from synthflow.chat.magic_words import detect_magic_word
        assert detect_magic_word("") is None


# – 3. SynSession –

class TestSynSession:
    def test_empty_on_init(self):
        from synthflow.chat.session import SynSession
        assert len(SynSession()) == 0

    def test_add_increases_length(self):
        from synthflow.chat.session import SynSession
        s = SynSession()
        s.add("user msg", "assistant msg")
        assert len(s) == 1

    def test_multiple_adds(self):
        from synthflow.chat.session import SynSession
        s = SynSession()
        s.add("a", "b")
        s.add("c", "d")
        assert len(s) == 2

    def test_clear_resets(self):
        from synthflow.chat.session import SynSession
        s = SynSession()
        s.add("x", "y")
        s.clear()
        assert len(s) == 0

    def test_turns_returns_list(self):
        from synthflow.chat.session import SynSession
        s = SynSession()
        s.add("u", "a")
        assert isinstance(s.turns, list)
        assert s.turns[0].user == "u"
        assert s.turns[0].assistant == "a"

    def test_as_messages_format(self):
        from synthflow.chat.session import SynSession
        s = SynSession()
        s.add("hello", "hi")
        msgs = s.as_messages()
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"


# – 4. SynState –

class TestSynState:
    def test_initial_state_is_collecting(self):
        from synthflow.chat.state_machine import SynState
        assert SynState().state == "COLLECTING"

    def test_initial_config_is_none(self):
        from synthflow.chat.state_machine import SynState
        assert SynState().config is None

    def test_transition_to_executing(self):
        from synthflow.chat.state_machine import SynState
        s = SynState()
        s.transition("EXECUTING")
        assert s.state == "EXECUTING"

    def test_transition_back_to_collecting(self):
        from synthflow.chat.state_machine import SynState
        s = SynState()
        s.transition("EXECUTING")
        s.transition("COLLECTING")
        assert s.state == "COLLECTING"

    def test_invalid_transition_raises(self):
        from synthflow.chat.state_machine import SynState
        from synthflow.exceptions import SynConfigError
        s = SynState()
        with pytest.raises(SynConfigError):
            s.transition("INVALID")

    def test_update_config(self):
        from synthflow.chat.state_machine import SynState
        from synthflow.parser import SynConfig
        s = SynState()
        cfg = SynConfig(domain="industrial")
        s.update_config(cfg)
        assert s.config.domain == "industrial"

    def test_reset_clears_config(self):
        from synthflow.chat.state_machine import SynState
        from synthflow.parser import SynConfig
        s = SynState()
        s.update_config(SynConfig())
        s.reset()
        assert s.config is None

    def test_reset_returns_to_collecting(self):
        from synthflow.chat.state_machine import SynState
        s = SynState()
        s.transition("EXECUTING")
        s.reset()
        assert s.state == "COLLECTING"

    def test_reset_clears_session(self):
        from synthflow.chat.state_machine import SynState
        s = SynState()
        s.session.add("x", "y")
        s.reset()
        assert len(s.session) == 0

    def test_is_ready_false_without_config(self):
        from synthflow.chat.state_machine import SynState
        assert not SynState().is_ready()

    def test_is_ready_true_with_config(self):
        from synthflow.chat.state_machine import SynState
        from synthflow.parser import SynConfig
        s = SynState()
        s.update_config(SynConfig())
        assert s.is_ready()


# – 5. Override detection –

class TestOverrideDetection:
    def test_model_override(self):
        from synthflow.chat.overrides import detect_override
        result = detect_override("change model to TimeVAE")
        assert result == {"model": "TimeVAE"}

    def test_sampling_rate_hz_override(self):
        from synthflow.chat.overrides import detect_override
        result = detect_override("sampling rate is 500 Hz")
        assert result is not None
        assert "sampling_rate_hz" in result
        assert result["sampling_rate_hz"] == pytest.approx(500.0)

    def test_n_samples_override(self):
        from synthflow.chat.overrides import detect_override
        result = detect_override("generate 2000 samples")
        assert result is not None
        assert result.get("n_samples") == 2000

    def test_device_gpu_maps_to_cuda(self):
        from synthflow.chat.overrides import detect_override
        result = detect_override("run on gpu")
        assert result is not None
        assert result.get("device") == "cuda"

    def test_no_override_returns_none(self):
        from synthflow.chat.overrides import detect_override
        assert detect_override("accelerometer from wind turbine") is None


# – 6. Guardrails –

class TestGuardrails:
    def test_repair_json_strips_markdown(self):
        from synthflow.parser.guardrails import repair_json
        text = '```json\n{"domain": "industrial"}\n```'
        result = repair_json(text)
        assert json.loads(result)["domain"] == "industrial"

    def test_repair_json_extracts_from_prose(self):
        from synthflow.parser.guardrails import repair_json
        text = 'Here is the config: {"domain": "iot"} as requested.'
        result = repair_json(text)
        assert json.loads(result)["domain"] == "iot"

    def test_parse_llm_response_valid_json(self):
        from synthflow.parser.guardrails import parse_llm_response
        result = parse_llm_response('{"domain": "industrial", "sampling_rate_hz": 500}')
        assert result["domain"] == "industrial"
        assert result["sampling_rate_hz"] == 500

    def test_parse_llm_response_removes_invalid_fields(self):
        from synthflow.parser.guardrails import parse_llm_response
        result = parse_llm_response('{"domain": "industrial", "fake_field": "bad"}')
        assert "fake_field" not in result
        assert "domain" in result

    def test_parse_llm_response_returns_empty_on_bad_json(self):
        from synthflow.parser.guardrails import parse_llm_response
        result = parse_llm_response("this is not json at all")
        assert result == {}


# – 7. SynParser (mocked API) –

class TestSynParser:
    def test_parse_returns_dict(self):
        from synthflow.parser import SynParser
        parser = SynParser(api_key="sk-ant-test")
        fake_resp = mock_anthropic_response({"domain": "industrial"})
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = fake_resp
            result = parser.parse("accelerometer from a wind turbine")
        assert isinstance(result, dict)

    def test_parse_extracts_domain(self):
        from synthflow.parser import SynParser
        parser = SynParser(api_key="sk-ant-test")
        fake_resp = mock_anthropic_response({
            "domain": "industrial",
            "sensor_type": "accelerometer",
        })
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = fake_resp
            result = parser.parse("accelerometer from a wind turbine")
        assert result.get("domain") == "industrial"

    def test_parse_returns_empty_on_api_failure(self):
        from synthflow.parser import SynParser
        parser = SynParser(api_key="sk-ant-test")
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.side_effect = Exception("API down")
            result = parser.parse("some message")
        assert result == {}


# – 8. SynFlow manual mode –

class TestSynFlowManualMode:
    def test_manual_generate_returns_syn_result(self, tmp_path):
        from synthflow import SynFlow
        from synthflow.parser import SynConfig
        from synthflow.output import SynResult
        csv_path = make_csv_file(tmp_path)
        sf = SynFlow(
            mode="manual",
            config=SynConfig(
                domain="industrial",
                model="GaussianProcess",
                n_samples=50,
            ),
            data=csv_path,
        )
        result = sf.generate()
        assert isinstance(result, SynResult)

    def test_manual_generate_output_has_synthetic_prefix(self, tmp_path):
        from synthflow import SynFlow
        from synthflow.parser import SynConfig
        csv_path = make_csv_file(tmp_path)
        sf = SynFlow(
            mode="manual",
            config=SynConfig(model="GaussianProcess", n_samples=50),
            data=csv_path,
        )
        result = sf.generate()
        synth_cols = [c for c in result.data.columns if c.startswith("synthetic_")]
        assert len(synth_cols) > 0

    def test_manual_generate_zero_nans(self, tmp_path):
        from synthflow import SynFlow
        from synthflow.parser import SynConfig
        csv_path = make_csv_file(tmp_path)
        sf = SynFlow(
            mode="manual",
            config=SynConfig(model="GaussianProcess", n_samples=50),
            data=csv_path,
        )
        result = sf.generate()
        assert result.data.isna().sum().sum() == 0

    def test_manual_generate_correct_n_samples(self, tmp_path):
        from synthflow import SynFlow
        from synthflow.parser import SynConfig
        csv_path = make_csv_file(tmp_path)
        sf = SynFlow(
            mode="manual",
            config=SynConfig(model="GaussianProcess", n_samples=75),
            data=csv_path,
        )
        result = sf.generate()
        assert len(result.data) == 75

    def test_manual_result_has_quality_metrics(self, tmp_path):
        from synthflow import SynFlow
        from synthflow.parser import SynConfig
        csv_path = make_csv_file(tmp_path)
        sf = SynFlow(
            mode="manual",
            config=SynConfig(model="GaussianProcess", n_samples=50),
            data=csv_path,
        )
        result = sf.generate()
        assert "distribution_similarity" in result.quality_metrics

    def test_manual_chat_raises(self, tmp_path):
        from synthflow import SynFlow
        from synthflow.parser import SynConfig
        from synthflow.exceptions import SynConfigError
        sf = SynFlow(
            mode="manual",
            config=SynConfig(),
            data=make_csv_file(tmp_path),
        )
        with pytest.raises(SynConfigError):
            sf.chat("hello")


# – 9. SynFlow auto mode (mocked) –

class TestSynFlowAutoMode:
    def test_chat_returns_string(self):
        from synthflow import SynFlow
        sf = SynFlow(mode="auto", api_key="sk-ant-test")
        fake_resp = mock_anthropic_response({"domain": "industrial"})
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = fake_resp
            response = sf.chat("accelerometer from a wind turbine")
        assert isinstance(response, str)

    def test_chat_updates_state_config(self):
        from synthflow import SynFlow
        sf = SynFlow(mode="auto", api_key="sk-ant-test")
        fake_resp = mock_anthropic_response({"domain": "industrial"})
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = fake_resp
            sf.chat("accelerometer from a wind turbine")
        assert sf._state.config is not None
        assert sf._state.config.domain == "industrial"

    def test_chat_adds_to_session(self):
        from synthflow import SynFlow
        sf = SynFlow(mode="auto", api_key="sk-ant-test")
        fake_resp = mock_anthropic_response({"domain": "iot"})
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = fake_resp
            sf.chat("temperature sensor")
        assert len(sf._state.session) == 1

    def test_reset_magic_word(self):
        from synthflow import SynFlow
        from synthflow.parser import SynConfig
        sf = SynFlow(mode="auto", api_key="sk-ant-test")
        sf._state.update_config(SynConfig(domain="industrial"))
        response = sf.chat("reset")
        assert sf._state.config is None
        assert "cleared" in response.lower() or "start" in response.lower()

    def test_show_config_magic_word(self):
        from synthflow import SynFlow
        from synthflow.parser import SynConfig
        sf = SynFlow(mode="auto", api_key="sk-ant-test")
        sf._state.update_config(SynConfig(domain="iot"))
        response = sf.chat("show config")
        assert "iot" in response.lower()

    def test_generate_with_no_config_returns_message(self):
        from synthflow import SynFlow
        sf = SynFlow(mode="auto", api_key="sk-ant-test")
        response = sf.chat("generate")
        assert isinstance(response, str)
        assert sf._state.state == "COLLECTING"

    def test_generate_triggers_pipeline(self, tmp_path):
        from synthflow import SynFlow
        from synthflow.parser import SynConfig
        from synthflow.output import SynResult
        sf = SynFlow(
            mode="auto",
            api_key="sk-ant-test",
            data=make_csv_file(tmp_path),
        )
        sf._state.update_config(SynConfig(
            model="GaussianProcess", n_samples=50
        ))
        result = sf.chat("generate")
        assert isinstance(result, SynResult)
        assert sf._state.state == "COLLECTING"

    def test_override_applied_mid_conversation(self):
        from synthflow import SynFlow
        from synthflow.parser import SynConfig
        sf = SynFlow(mode="auto", api_key="sk-ant-test")
        sf._state.update_config(SynConfig(model="TimeGAN"))
        sf.chat("change model to TimeVAE")
        assert sf._state.config.model == "TimeVAE"

    def test_multiple_chat_messages_accumulate(self):
        from synthflow import SynFlow
        sf = SynFlow(mode="auto", api_key="sk-ant-test")
        fake_resp1 = mock_anthropic_response({"domain": "industrial"})
        fake_resp2 = mock_anthropic_response({"sensor_type": "accelerometer"})
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.side_effect = [
                fake_resp1, fake_resp2
            ]
            sf.chat("wind turbine data")
            sf.chat("it is an accelerometer")
        assert sf._state.config.domain == "industrial"
        assert sf._state.config.sensor_type == "accelerometer"
