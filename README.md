# synthflow

Synthetic sensor data generation with automatic imputation and model selection.

## Install

pip install synthflow

## Quickstart

### Auto mode (Claude API)
from synthflow import SynFlow

sf = SynFlow(mode="auto", api_key="sk-ant-...")
sf.chat("accelerometer from a wind turbine, 500Hz, dropout during storms")
sf.chat("generate")

### Manual mode
from synthflow import SynFlow
from synthflow.parser import SynConfig

config = SynConfig(
    domain="industrial",
    sensor_type="accelerometer",
    sampling_rate_hz=500,
    backend="tsgm",
    model="TimeVAE",
)
sf = SynFlow(mode="manual", config=config, data="sensor_data.csv")
result = sf.generate()
print(result.data.head())
