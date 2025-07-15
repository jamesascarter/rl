#!/usr/bin/env python3
"""
Simple inference for LoRA fine-tuned Qwen model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def generate_summary(text: str, lora_path: str = "./lora-output/best-model") -> str:
    """
    Generate a summary using your LoRA fine-tuned model
    
    Args:
        text: Text to summarize
        lora_path: Path to your LoRA weights directory
        
    Returns:
        Generated summary
    """
    # Load base model and tokenizer
    model_name = "Qwen/Qwen1.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, lora_path)
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Create prompt
    prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract summary
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = generated_text.split("Summary:")[-1].strip()
    
    return summary


if __name__ == "__main__":
    # Test with your text
    text = """
    SUBREDDIT: r/relationships TITLE: The girl [26 F] I [22 M] have been seeing for a month didn't respond to me at all yesterday while hanging out with a friend [~30? M]. POST: She gets terrible service while at her house, but I texted her 3 times yesterday, 4-5 hours apart. She didn't call me until early this morning and left a voicemail that she was busy all day with a friend who showed up out of the blue. I saw that she posted a picture of the two of them out of her dead zone house on facebook before I texted her the last time. I don't mind that she hangs out with friends, and I know it's pretty early in the relationship, but am I wrong to be a little annoyed that she didn't respond until 24 hours after my first text? TL;DR:
    """
    
    summary = generate_summary(text)
    print(f"Summary: {summary}") 