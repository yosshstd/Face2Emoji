import torch
#import timm
from transformers import ViTModel, ViTForImageClassification, CLIPVisionModelWithProjection
from torch import nn
from peft import LoraConfig, get_peft_model


class ViT(nn.Module):
    def __init__(self, model_name: str, trainable: bool = False, num_classes: int = 7, lora: bool = False) -> None:
        super().__init__()

        #self.model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self.model.init_weights()
        self.head = nn.Linear(self.model.config.hidden_size, num_classes)
        #self.model =  timm.create_model(model_name, pretrained=True, num_classes=0)
        
        self.dropout = nn.Dropout(0.1)

        for param in self.model.parameters():
            param.requires_grad = trainable

        if lora:
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()
            config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=['q_proj', 'v_proj'],
                lora_dropout=0.1,
                bias="none"
            )
            self.model = get_peft_model(self.model, config)
        
        print_trainable_parameters(self.model) 

    def forward(self, x):
        x = self.model(x)
        logits = self.head(x.image_embeds)
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