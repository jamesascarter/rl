#!/usr/bin/env python3
"""
Simple training loop with wandb integration for LoRA fine-tuning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import wandb
import argparse
import os
from tqdm import tqdm
from models import QwenSftModel


class TLDRDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Format prompt
        prompt = f"Summarize the following text:\n\n{example['prompt']}\n\nSummary:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt + example['completion'],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (mask prompt tokens with -100)
        labels = inputs["input_ids"].clone()
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        prompt_length = prompt_tokens["input_ids"].shape[1]
        labels[:, :prompt_length] = -100
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }


def train(
    model_name="Qwen/Qwen1.5-0.5B",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    batch_size=16,  # Increased from 4
    learning_rate=1e-4,  # Increased from 5e-5
    num_epochs=3,
    max_length=512,
    warmup_steps=100,
    save_steps=500,
    eval_steps=500,
    output_dir="./lora-output",
    wandb_project="qwen-lora-tldr",
    wandb_run_name=None
):
    """
    Training loop with wandb integration
    """
    
    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "model_name": model_name,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_length": max_length,
            "warmup_steps": warmup_steps
        }
    )
    
    # Initialize model
    print("Loading model...")
    model = QwenSftModel(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("trl-lib/tldr")
    
    # Create train/validation datasets
    train_dataset = dataset["train"].select(range(min(1000, len(dataset["train"]))))
    val_dataset = dataset["validation"].select(range(min(100, len(dataset["validation"]))))
    
    # Create PyTorch datasets
    train_dataset = TLDRDataset(train_dataset, model.tokenizer, max_length)
    val_dataset = TLDRDataset(val_dataset, model.tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup training
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Explicitly use GPU 0
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU - this will be very slow!")
    
    model.to(device)
    
    # Verify model is on correct device
    print(f"Model device: {next(model.parameters()).device}")
    
    # Test GPU computation
    if torch.cuda.is_available():
        test_tensor = torch.randn(2, 2).to(device)
        result = test_tensor @ test_tensor.T
        print(f"GPU test successful: {result.shape}")
    else:
        print("Running on CPU - training will be very slow!")
    
    # Optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in train_pbar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to wandb
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "global_step": global_step
            })
            
            # Save checkpoint
            if global_step % save_steps == 0:
                model.save_lora_weights(f"{output_dir}/checkpoint-{global_step}")
                wandb.log({"checkpoint_saved": global_step})
            
            # Evaluation
            if global_step % eval_steps == 0:
                val_loss = evaluate(model, val_loader, device, criterion)
                wandb.log({
                    "val_loss": val_loss,
                    "global_step": global_step
                })
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_lora_weights(f"{output_dir}/best-model")
                    wandb.log({"best_val_loss": val_loss})
                
                model.train()
        
        # Epoch summary
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")
        
        # Final evaluation for epoch
        val_loss = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "epoch_train_loss": avg_train_loss,
            "epoch_val_loss": val_loss
        })
    
    # Save final model
    model.save_lora_weights(f"{output_dir}/final-model")
    print(f"Training completed! Model saved to {output_dir}")
    
    # Close wandb
    wandb.finish()


def evaluate(model, val_loader, device, criterion):
    """
    Evaluate model on validation set
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train Qwen with LoRA and wandb")
    parser.add_argument("--model_name", default="Qwen/Qwen1.5-0.5B", help="Model name")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")  # Increased from 4
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")  # Increased from 5e-5
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--output_dir", default="./lora-output", help="Output directory")
    parser.add_argument("--wandb_project", default="qwen-lora-tldr", help="Wandb project name")
    parser.add_argument("--wandb_run_name", default=None, help="Wandb run name")
    
    args = parser.parse_args()
    
    train(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )


if __name__ == "__main__":
    main() 