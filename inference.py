import torch
from models import QwenSftModel
from typing import Optional, Dict, Any


class QwenInference:
    def __init__(self, model_path: Optional[str] = None, model_name: str = "Qwen/Qwen1.5-0.5B"):
        """
        Initialize QwenInference for text generation and summarization
        
        Args:
            model_path: Path to fine-tuned model (if None, uses base model)
            model_name: HuggingFace model name (used if model_path is None)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        if model_path:
            self.model = QwenSftModel(model_name)
            self.model.load_lora_weights(model_path)  # Use the existing method
            print(f"Loaded fine-tuned model from {model_path}")
        else:
            self.model = QwenSftModel(model_name)
            print(f"Loaded base model: {model_name}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> str:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return
            
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.model.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        # Generate
        with torch.no_grad():
            outputs = self.model.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.model.tokenizer.eos_token_id,
                eos_token_id=self.model.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode output
        generated_text = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        print("before")
        return generated_text
        print("after")
    
    def generate_summary(
        self,
        text: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a summary for given text
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of generated summary
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated summary
        """
        prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
        
        summary = self.generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        
        return summary
    
    def batch_generate(
        self,
        prompts: list,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> list:
        """
        Generate text for multiple prompts in batch
        
        Args:
            prompts: List of input prompts
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            List of generated texts
        """
        results = []
        
        for prompt in prompts:
            result = self.generate_text(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_name": self.model.model.config._name_or_path,
            "vocab_size": self.model.tokenizer.vocab_size,
            "max_position_embeddings": self.model.model.config.max_position_embeddings
        }


# Example usage
if __name__ == "__main__":
    # Initialize inference
    inference = QwenInference(model_path="./lora-output/best-model")
    
    # Test text generation
    prompt = "SUBREDDIT: r/relationships TITLE: The girl [26 F] I [22 M] have been seeing for a month didn't respond to me at all yesterday while hanging out with a friend [~30? M]. POST: She gets terrible service while at her house, but I texted her 3 times yesterday, 4-5 hours apart. She didn't call me until early this morning and left a voicemail that she was busy all day with a friend who showed up out of the blue. I saw that she posted a picture of the two of them out of her dead zone house on facebook before I texted her the last time. I don't mind that she hangs out with friends, and I know it's pretty early in the relationship, but am I wrong to be a little annoyed that she didn't respond until 24 hours after my first text? TL;DR:"
    generated = inference.generate_text(prompt, max_length=50)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    print()
    
    # Test summarization
    text = """
    Girl I'm seeing didn't respond to my texts while hanging out with a friend, what should I do, if anything?
    """
    summary = inference.generate_summary(text)
    print(f"Original text: {text.strip()}")
    print(f"Summary: {summary}")
    print()
    
    # Get model info
    info = inference.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}") 