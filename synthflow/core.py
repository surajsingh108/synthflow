"""
SynFlow – main entry point for the synthflow package.

Two modes:
  "auto"   – natural language via Claude API chat interface
  "manual" – direct SynConfig object, generate() called immediately
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from synthflow.exceptions import SynConfigError, SynBackendError
from synthflow.parser.schema import SynConfig


class SynFlow:
    """
    Main synthflow interface.

    Auto mode:
        sf = SynFlow(mode="auto", api_key="sk-ant-...")
        sf.chat("accelerometer from a wind turbine, 500Hz")
        sf.chat("change model to TimeVAE")
        result = sf.chat("generate")   # returns SynResult

    Manual mode:
        sf = SynFlow(mode="manual", config=SynConfig(...), data="data.csv")
        result = sf.generate()
    """

    VERSION = "0.1.0"

    def __init__(
        self,
        mode: str = "auto",
        api_key: str | None = None,
        config: SynConfig | dict | None = None,
        data: str | Path | pd.DataFrame | None = None,
    ):
        if mode not in ("auto", "manual"):
            raise SynConfigError(
                f"Invalid mode: '{mode}'",
                detail="mode must be 'auto' or 'manual'",
            )

        if mode == "auto" and api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise SynConfigError(
                    "api_key is required for auto mode",
                    detail="pass api_key= or set ANTHROPIC_API_KEY env var",
                )

        if mode == "manual" and config is None:
            raise SynConfigError(
                "config is required for manual mode",
                detail="pass a SynConfig instance",
            )

        self.mode = mode
        self.api_key = api_key
        self.data = data
        self._result = None
        self.config = config

        # normalise config for internal use
        if isinstance(config, dict):
            self._normalized_config = SynConfig(**config)
        elif isinstance(config, SynConfig):
            self._normalized_config = config
        else:
            self._normalized_config = None

        # auto mode state machine
        if mode == "auto":
            from synthflow.chat.state_machine import SynState
            self._state = SynState()
        else:
            self._state = None

    def chat(self, message: str):
        """
        Send a natural language message (auto mode only).

        Returns:
            str response for most messages.
            SynResult when "generate" is triggered.

        Raises:
            SynConfigError if called in manual mode.
        """
        if self.mode != "auto":
            raise SynConfigError(
                "chat() is only available in auto mode.",
                detail="Use generate() directly in manual mode.",
            )

        from synthflow.chat.magic_words import detect_magic_word
        from synthflow.chat.overrides import detect_override
        from synthflow.parser.lm_parser import SynParser

        magic = detect_magic_word(message)

        # – magic: generate –
        if magic == "generate":
            if self._state.config is None:
                return (
                    "No config set yet. Describe your sensor data first, "
                    "then say 'generate'."
                )
            self._state.transition("EXECUTING")
            try:
                result = self._run_pipeline(self._state.config)
                self._result = result
                self._state.transition("COLLECTING")
                return result
            except Exception as exc:
                self._state.transition("COLLECTING")
                raise SynBackendError(
                    "Pipeline failed during generation.",
                    detail=str(exc),
                ) from exc

        # – magic: reset –
        if magic == "reset":
            self._state.reset()
            return "Config cleared. Describe your sensor data to start again."

        # – magic: show config –
        if magic == "show config":
            if self._state.config is None:
                return "No config set yet."
            return self._state.config.summary()

        # – magic: explain –
        if magic == "explain":
            if self._state.config is None:
                return "No config to explain yet."
            parser = SynParser(api_key=self.api_key)
            return parser.explain(self._state.config)

        # – direct field override –
        override = detect_override(message)
        if override and self._state.config is not None:
            try:
                new_config = self._state.config.patch(**override)
                self._state.update_config(new_config)
                field = list(override.keys())[0]
                val = list(override.values())[0]
                response = f"Updated {field} → {val}.\n\n{new_config.summary()}"
                self._state.session.add(message, response)
                return response
            except Exception:
                pass  # fall through to LM parse

        # – LM parse –
        parser = SynParser(api_key=self.api_key)
        patch = parser.parse(message, self._state.config)

        if patch:
            if self._state.config is None:
                try:
                    new_config = SynConfig(**patch)
                except Exception:
                    new_config = SynConfig()
            else:
                try:
                    new_config = self._state.config.patch(**patch)
                except Exception:
                    new_config = self._state.config
            self._state.update_config(new_config)
            response = (
                f"Got it. Here's my plan:\n"
                f"{new_config.summary()}\n\n"
                "Does this look right? Say 'generate' to start, "
                "or keep refining."
            )
        else:
            response = (
                "I couldn't extract config from that. "
                "Try describing the sensor type, domain, or sampling rate."
            )

        self._state.session.add(message, response)
        return response

    def generate(self):
        """
        Run the full synthesis pipeline (both modes).

        In auto mode: uses the config built via chat().
        In manual mode: uses the SynConfig passed at init.

        Returns:
            SynResult
        """
        if self.mode == "auto":
            if self._state.config is None:
                raise SynConfigError(
                    "No config set. Use chat() to describe your data first.",
                )
            config = self._state.config
        else:
            if self._normalized_config is None or self.data is None:
                raise NotImplementedError("generate() requires config and data")
            config = self._normalized_config

        result = self._run_pipeline(config)
        self._result = result
        return result

    # – pipeline –

    def _run_pipeline(self, config: SynConfig):
        """
        Full synthesis pipeline:
          1. Ingest data
          2. Impute missing values
          3. Select model (router)
          4. Synthesize (TSGM backend)
          5. Compute quality metrics
          6. Return SynResult
        """
        from synthflow.ingestor import SynIngestor
        from synthflow.imputer import SynImputer
        from synthflow.router import SynRouter
        from synthflow.backends import TsgmBackend
        from synthflow.output import SynResult
        from synthflow.output.quality import compute_all_metrics
        from synthflow.output.report import SynReport

        # 1. ingest
        if self.data is None:
            raise SynConfigError(
                "No data source provided.",
                detail="Pass data= when creating SynFlow.",
            )
        load_result = SynIngestor().load(self.data)

        # 2. impute
        imp_result = SynImputer().impute(
            df=load_result.data,
            missing_pattern=config.missing_pattern,
            imputation_strategy=config.imputation_strategy,
            imputation_overrides=config.imputation_overrides,
            signal_cols=load_result.signal_cols,
        )

        # 3. route
        fs = load_result.sampling_rate_hz or config.sampling_rate_hz
        selection = SynRouter().select(
            domain=config.domain,
            n_rows=load_result.n_rows,
            sampling_rate_hz=fs,
            n_signal_cols=len(load_result.signal_cols),
            preferred_model=config.model,
            default_batch_size=config.batch_size,
        )

        # 4. synthesize
        synth_df = TsgmBackend().run(
            df=imp_result.data,
            signal_cols=load_result.signal_cols,
            timestamp_col=load_result.timestamp_col,
            model_name=selection.model,
            n_samples=config.n_samples,
            epochs=config.training_epochs,
            batch_size=selection.batch_size,
            random_seed=config.random_seed,
            sampling_rate_hz=fs,
            seq_len=config.seq_len,
            data_type=config.data_type,
            auto_bounds=config.auto_bounds,
            column_bounds=dict(config.column_bounds),
        )

        # 5. quality
        quality = compute_all_metrics(
            imp_result.data, synth_df, load_result.signal_cols
        )

        # 6. column descriptions
        col_desc = SynReport.build_column_descriptions(
            synth_df, load_result.signal_cols
        )

        return SynResult(
            data=synth_df,
            config=config.model_dump(),
            imputation_report=imp_result.as_dict(),
            quality_metrics=quality,
            column_descriptions=col_desc,
            run_id=SynReport.make_run_id(),
            signal_cols=load_result.signal_cols,
            timestamp_col=load_result.timestamp_col,
            source=load_result.source,
            model=selection.model,
            backend="tsgm",
            sampling_rate_hz=fs,
        )

    def __repr__(self) -> str:
        return (
            f"SynFlow(mode='{self.mode}', "
            f"data='{self.data}', "
            f"config={'set' if self.config else 'not set'})"
        )
