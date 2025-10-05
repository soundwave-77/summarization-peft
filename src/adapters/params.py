from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from src.adapters.constants import TASK_TYPE


class BaseParams(ABC, BaseModel, frozen=True):
    task_type: str = Field(default=TASK_TYPE, description="Task type")
    inference_mode: bool = Field(
        default=False,
        description="If True, disables training and uses the adapter only for inference",
    )

    @abstractmethod
    def _experiment_name() -> str:
        pass


class LoraParams(BaseParams):
    r: int = Field(default=8, description="LoRA rank")
    lora_alpha: int = Field(default=8, description="LoRA alpha")
    lora_dropout: float = Field(default=0, description="Dropout probability for LoRA")
    use_dora: bool = Field(default=False, description="Use DoRA")
    target_modules: list[str] | str | None = Field(
        default=None,
        description="The names of the modules to apply the adapter to",
    )

    def _experiment_name(self) -> str:
        dora = "DoRA" if self.use_dora else "LoRA"
        if self.target_modules is None:
            tm = ""
        elif isinstance(self.target_modules, str):
            tm = f"_modules_{self.target_modules}"
        else:
            tm = f'_modules_{"_".join(sorted(self.target_modules))}'
        return f"Lora_{dora}_r{self.r}_a{self.lora_alpha}_d{self.lora_dropout}{tm}"


class PrefixTuningParams(BaseParams):
    num_virtual_tokens: int = Field(
        default=8,
        description="Number of virtual tokens to prepend to the input sequence",
    )
    encoder_hidden_size: int = Field(
        default=128,
        description="The hidden size of the prompt encoder",
    )
    prefix_projection: bool = Field(
        default=False,
        description="Whether to use a two-layer MLP to project prefix embeddings to hidden size",
    )

    def _experiment_name(self) -> str:
        return f"PrefixTuning_vt{self.num_virtual_tokens}_h{self.encoder_hidden_size}_proj_{self.prefix_projection}"


class PromptTuningParams(BaseParams):
    num_virtual_tokens: int = Field(
        default=2,
        description="Number of virtual tokens to prepend to the input sequence",
    )
    prompt_tuning_init: Literal["RANDOM", "TEXT"] = Field(
        default="RANDOM",
        description="How to initialize the prompt embeddings: 'RANDOM' or from input 'TEXT'",
    )
    prompt_tuning_init_text: str | None = Field(
        default=None,
        description="The text to initialize the prompt embedding",
    )

    def _experiment_name(self) -> str:
        name = f"PromptTuning_vt{self.num_virtual_tokens}_init_{self.prompt_tuning_init}"
        if self.prompt_tuning_init == "TEXT" and self.prompt_tuning_init_text:
            init_text = self.prompt_tuning_init_text.replace(" ", "-")
            name += f"_{init_text}"
        return name


class PTuningParams(BaseParams):
    num_virtual_tokens: int = Field(
        default=8,
        description="Number of virtual tokens to prepend to the input sequence",
    )
    encoder_reparameterization_type: Literal["MLP", "LSTM"] = Field(
        default="MLP",
        description="The type of reparameterization to use",
    )
    encoder_hidden_size: int = Field(
        default=128,
        description="The hidden size of the prompt encoder",
    )
    encoder_num_layers: int = Field(
        default=2,
        description="The number of layers of the prompt encoder",
    )

    def _experiment_name(self) -> str:
        return f"PTuning_vt{self.num_virtual_tokens}_{self.encoder_reparameterization_type}_h{self.encoder_hidden_size}_L{self.encoder_num_layers}"
