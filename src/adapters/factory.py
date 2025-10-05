import json
import os

from peft import (
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
)

from src.adapters.constants import ADAPTER_CONFIG_DIR
from src.adapters.params import (
    BaseParams,
    LoraParams,
    PrefixTuningParams,
    PromptTuningParams,
    PTuningParams,
)


class AdapterFactory:
    def __init__(self, adapter_config_dir: str = ADAPTER_CONFIG_DIR):
        self.adapter_config_dir = adapter_config_dir
        self.adapter_params = None

    def create_adapter(self, adapter_name: str):
        match adapter_name:
            case PeftType.LORA:
                config_class = LoraConfig
                params_class = LoraParams
            case PeftType.PREFIX_TUNING:
                config_class = PrefixTuningConfig
                params_class = PrefixTuningParams
            case PeftType.PROMPT_TUNING:
                config_class = PromptTuningConfig
                params_class = PromptTuningParams
            case PeftType.P_TUNING:
                config_class = PromptEncoderConfig
                params_class = PTuningParams
            case _:
                raise ValueError("Unknown adapter")

        adapter_config = self._load_config(adapter_name)

        self.adapter_params = self._validate_config(adapter_config, params_class)
        
        return config_class(**adapter_config)

    def _load_config(self, adapter_name: str) -> dict:
        adapter_config_path = os.path.join(
            self.adapter_config_dir,
            f"{adapter_name.lower()}.json",
        )

        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        
        return adapter_config

    def _validate_config(self, data: dict, model: type[BaseParams]):
        return model.model_validate(data)
    
    def experiment_name(self) -> str:
        return self.adapter_params._experiment_name()
        
