from config import ExpConfig
from model.bert_wrapper import BertForPEFT

config = ExpConfig(peft_type="lora")  # adapter | prefix | ptuning
model = BertForPEFT(config)

print("Trainable params:")
for n, p in model.named_parameters():
    if p.requires_grad:
        print(n, p.numel())
