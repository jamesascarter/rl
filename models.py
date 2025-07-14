import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


class QwenSftModel(nn.Module):
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen1.5-0.5B",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: list = None
    ):
        """
        Initialize QwenSftModel with LoRA on attention layers
        
        Args:
            model_name: HuggingFace model name (default: small Qwen model)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: List of module names to apply LoRA to
        """
        super().__init__()
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Freeze all weights in the base model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Set default target modules for Qwen attention layers
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none"
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Print trainable parameters info
        self.model.print_trainable_parameters()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for training (optional)
            
        Returns:
            Model outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def save_lora_weights(self, path: str):
        """
        Save only the LoRA weights
        
        Args:
            path: Directory to save LoRA weights
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"LoRA weights saved to {path}")
    
    def load_lora_weights(self, path: str):
        """
        Load LoRA weights
        
        Args:
            path: Directory containing LoRA weights
        """
        self.model = self.model.from_pretrained(path)
        print(f"LoRA weights loaded from {path}")
    
    def get_trainable_parameters(self):
        """
        Get information about trainable parameters
        
        Returns:
            Dictionary with parameter information
        """
        trainable_params = 0
        all_param = 0
        
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            "trainable_params": trainable_params,
            "all_params": all_param,
            "trainable_percentage": 100 * trainable_params / all_param
        }
