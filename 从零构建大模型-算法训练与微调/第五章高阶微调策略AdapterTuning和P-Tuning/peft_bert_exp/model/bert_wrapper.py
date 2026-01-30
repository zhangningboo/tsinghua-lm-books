import torch
import torch.nn as nn
from transformers import BertModel

from model.lora import LoRALinear
from model.adapter import Adapter
from model.prefix import PrefixEncoder
from model.ptuning import PromptEmbedding

class BertForPEFT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.peft_type = config.peft_type

        # 冻结 backbone
        for p in self.bert.parameters():
            p.requires_grad = False

        if config.peft_type == "lora":
            from transformers.models.bert.modeling_bert import BertSelfAttention

            for layer in self.bert.encoder.layer:
                attn: BertSelfAttention = layer.attention.self # type: ignore

                attn.query = LoRALinear( # type: ignore
                    attn.query,
                    config.lora_r,
                    config.lora_alpha
                )
                attn.value = LoRALinear( # type: ignore
                    attn.value,
                    config.lora_r,
                    config.lora_alpha
                )

        elif config.peft_type == "adapter":
            self.adapters = nn.ModuleList([
                Adapter(config.hidden_size, config.adapter_dim)
                for _ in self.bert.encoder.layer
            ])

        elif config.peft_type == "prefix":
            self.prefix = PrefixEncoder(config.prefix_len, config.hidden_size)

        elif config.peft_type == "ptuning":
            self.prompt = PromptEmbedding(config.prompt_len, config.hidden_size)

    def forward(self, input_ids, attention_mask):
        bsz = input_ids.size(0)

        if self.peft_type == "ptuning":
            emb = self.bert.embeddings(input_ids)
            prompt = self.prompt(bsz)
            emb = torch.cat([prompt, emb], dim=1)
            attention_mask = torch.cat(
                [torch.ones(bsz, prompt.size(1), device=attention_mask.device),
                 attention_mask],
                dim=1
            )
            outputs = self.bert(inputs_embeds=emb, attention_mask=attention_mask)

        else:
            outputs = self.bert(input_ids, attention_mask)

        hidden = outputs.last_hidden_state

        if self.peft_type == "adapter":
            for i, adapter in enumerate(self.adapters):
                hidden = adapter(hidden)

        cls = hidden[:, 0]
        return self.classifier(cls)
