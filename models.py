import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

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
        self.model = PeftModel.from_pretrained(self.model, path)  # Fixed this line
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
class RewardModel(nn.Module):
    def __init__(self, sft_model_path: str = "./lora-output/best-model", model_name: str = "Qwen/Qwen1.5-0.5B"):
        super().__init__()
        
        
        # Load tokenizer separately first
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_hidden_states=True
        )
        self.hidden_size = self.base_model.config.hidden_size

        # Load LoRA weights directly
        self.base_model = PeftModel.from_pretrained(self.base_model, sft_model_path)

        # Freeze ALL parameters in the base model (including LoRA weights)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Reward head: scalar output from last token (only this will be trained)
        self.reward_head = nn.Linear(self.hidden_size, 1)
        
        # Print trainable parameters info
        print("=== REWARD MODEL PARAMETERS ===")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        print("Only reward head is trainable!")

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Get the last hidden layer (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.hidden_states[-1]

        # Extract the hidden state of the last non-padding token
        # Assuming padding is on the left (HuggingFace default for causal models)
        lengths = attention_mask.sum(dim=1) - 1  # (batch_size,)
        last_token_hidden = last_hidden_state[range(last_hidden_state.size(0)), lengths]

        # Compute scalar reward
        rewards = self.reward_head(last_token_hidden).squeeze(-1)  # (batch_size,)
        return rewards
    
    def save_reward_model(self, path: str):
        """
        Save the reward head and tokenizer
        """
        os.makedirs(path, exist_ok=True)
        
        # Save the reward head
        torch.save(self.reward_head.state_dict(), f"{path}/reward_head.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        print(f"Reward head saved to {path}")

    def load_reward_model(self, path: str):
        """
        Load the reward head
        """
        # Load reward head
        reward_head_path = f"{path}/reward_head.pt"
        if os.path.exists(reward_head_path):
            self.reward_head.load_state_dict(torch.load(reward_head_path))
            print(f"Reward head loaded from {path}")
        else:
            print(f"Warning: No reward head found at {path}")