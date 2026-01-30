from dataclasses import dataclass

@dataclass
class ExpConfig:
    model_name: str = "../../bert-base-uncased"
    num_labels: int = 2
    peft_type: str = "lora"  # lora | adapter | prefix | ptuning

    # 通用
    hidden_size: int = 768

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 32

    # Adapter
    adapter_dim: int = 64

    # Prefix
    prefix_len: int = 10

    # P-Tuning
    prompt_len: int = 10

    lr: float = 2e-4
