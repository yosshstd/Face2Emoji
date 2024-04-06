import torch
#import timm
from transformers import ViTModel
from torch import nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class ViT(nn.Module):
    def __init__(self, model_name: str, trainable: bool = False, num_classes: int = 7, lora: bool = False) -> None:
        super().__init__()

        self.vit = ViTModel.from_pretrained(model_name)
        self.head = nn.Linear(self.vit.config.hidden_size, num_classes)
        self.vit.init_weights()

        for param in self.vit.parameters():
            param.requires_grad = trainable

        if lora:
            self.vit.enable_input_require_grads()
            self.vit.gradient_checkpointing_enable()
            config = LoraConfig(
                r=8,
                lora_alpha=16, # lora_alpha should be 2 x r
                target_modules=['query', 'key', 'value', 'dense'],
                lora_dropout=0.1,
                bias="none"
            )
            self.vit = get_peft_model(self.vit, config)


        # '''実験用'''
        # SEQUENCE_LENGTH = 197
        # K = 8
        # for i in range(self.vit.config.num_hidden_layers):
        #     self.vit.encoder.layer[i].attention.attention.key = nn.Sequential(self.vit.encoder.layer[i].attention.attention.key, Transpose(), nn.Conv1d(768, 768, kernel_size=K, stride=K), Transpose())
        #     self.vit.encoder.layer[i].attention.attention.value = nn.Sequential(self.vit.encoder.layer[i].attention.attention.value, Transpose(), nn.Conv1d(768, 768, kernel_size=K, stride=K), Transpose())
        
        self.dropout = nn.Dropout(0.1)
            
            
        print_trainable_parameters(self.vit)


    def forward(self, x):
        last_hidden_states = self.vit(x).last_hidden_state
        logits = self.head(self.dropout(last_hidden_states[:, 0, :]))
        return logits

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )