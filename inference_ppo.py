#!/usr/bin/env python3
"""
Inference for PPO fine-tuned Qwen model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from models import RewardModel


def load_ppo_model(ppo_path: str = "./ppo-output", model_name: str = "Qwen/Qwen1.5-0.5B"):
    """
    Load the PPO-trained model
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load PPO LoRA weights
    ppo_model = PeftModel.from_pretrained(base_model, ppo_path)
    
    return ppo_model, tokenizer


def generate_summary_with_ppo(
    text: str, 
    ppo_model, 
    tokenizer, 
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """
    Generate summary using PPO-trained model
    """
    # Format prompt
    prompt = f"Summarize:\n{text.strip()}\nTL;DR:"
    
    # Tokenize
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    )
    
    # Move to device
    inputs = {k: v.to(ppo_model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = ppo_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    prompt_length = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
    summary = generated_text[prompt_length:].strip()
    
    return summary


def compare_models(
    text: str,
    sft_path: str = "./lora-output/best-model",
    ppo_path: str = "./ppo-output",
    reward_path: str = "./reward-output/best-model",
    model_name: str = "Qwen/Qwen1.5-0.5B"
):
    """
    Compare SFT vs PPO models and score with reward model
    """
    # Load models
    ppo_model, tokenizer = load_ppo_model(ppo_path, model_name)
    reward_model = RewardModel(sft_model_path=sft_path, model_name=model_name)
    reward_model.load_reward_model(reward_path)
    reward_model.eval()
    
    # Generate with SFT model
    sft_model, _ = load_ppo_model(sft_path, model_name)  # Reuse function
    sft_summary = generate_summary_with_ppo(text, sft_model, tokenizer)
    
    # Generate with PPO model
    ppo_summary = generate_summary_with_ppo(text, ppo_model, tokenizer)
    
    # Score both with reward model
    def score_summary(summary: str) -> float:
        full_text = f"Summarize:\n{text.strip()}\nTL;DR: {summary}"
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            reward = reward_model(**inputs)
        return reward.item()
    
    sft_score = score_summary(sft_summary)
    ppo_score = score_summary(ppo_summary)
    
    print("=== Model Comparison ===")
    print(f"Input text: {text[:100]}...")
    print(f"\nSFT Summary: {sft_summary}")
    print(f"SFT Reward: {sft_score:.4f}")
    print(f"\nPPO Summary: {ppo_summary}")
    print(f"PPO Reward: {ppo_score:.4f}")
    print(f"\nImprovement: {ppo_score - sft_score:.4f}")


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Artificial intelligence (AI) has rapidly evolved from a theoretical concept to a practical technology that impacts nearly every aspect of modern life. From virtual assistants like Siri and Alexa to recommendation systems on Netflix and Amazon, AI algorithms are working behind the scenes to enhance user experiences. In healthcare, AI is being used to diagnose diseases, predict patient outcomes, and even assist in surgical procedures. The financial sector relies on AI for fraud detection, algorithmic trading, and risk assessment. Transportation is being revolutionized by self-driving cars and smart traffic management systems. However, this rapid advancement also raises important questions about job displacement, privacy concerns, and the ethical implications of autonomous decision-making systems.
    """
    
    # Simple inference
    ppo_model, tokenizer = load_ppo_model()
    summary = generate_summary_with_ppo(sample_text, ppo_model, tokenizer)
    print(f"PPO Summary: {summary}")
    
    # Compare models
    compare_models(sample_text) 