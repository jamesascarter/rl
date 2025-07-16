import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import wandb
import argparse
import os
from tqdm import tqdm
from models import RewardModel


class ContrastiveDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Tokenize anchor, positive, negative
        prompt = example['prompt']
        chosen = example['chosen']
        rejected = example['rejected']

        chosen_text = f"Summarize the following text:\n\n{prompt}\n\nSummary: {chosen}"
        rejected_text = f"Summarize the following text:\n\n{prompt}\n\nSummary: {rejected}"

        chosen_tokens = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_tokens = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(),
        }

    def __len__(self):
        return len(self.dataset)


def train_preference(
    sft_model_path="./lora-output/best-model",  # Path to your trained SFT model
    model_name="Qwen/Qwen1.5-0.5B",
    batch_size=8,
    learning_rate=5e-5,
    num_epochs=3,
    max_length=512,
    warmup_steps=100,
    save_steps=500,
    eval_steps=500,
    output_dir="./reward-output",
    wandb_project="qwen-reward-preference",
    wandb_run_name=None,
    dataset_name="CarperAI/openai_summarize_comparisons"
):
    """
    Training loop with Bradley-Terry preference loss for RewardModel
    """

    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "sft_model_path": sft_model_path,
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_length": max_length,
            "warmup_steps": warmup_steps,
            "dataset_name": dataset_name
        }
    )
    
    # Initialize RewardModel using your trained SFT model
    print("Loading RewardModel with trained SFT model...")
    model = RewardModel(sft_model_path=sft_model_path, model_name=model_name)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_name)
    
    # Create train/validation datasets
    train_dataset = dataset["train"].select(range(min(100, len(dataset["train"]))))
    val_dataset = dataset["validation"].select(range(min(20, len(dataset["validation"]))))

    # Debug: Check dataset structure
    print("Debug: Checking dataset structure...")
    print(f"Dataset keys: {dataset.keys()}")
    print(f"Train dataset columns: {train_dataset.column_names}")
    print(f"First example keys: {train_dataset[0].keys()}")
    print(f"First example: {train_dataset[0]}")
    print()
    
    # Create PyTorch datasets
    train_dataset = ContrastiveDataset(train_dataset, model.tokenizer, max_length)
    val_dataset = ContrastiveDataset(val_dataset, model.tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("=== DATASET STATISTICS ===")
    print(f"Train dataset size: {len(train_dataset):,} examples")
    print(f"Validation dataset size: {len(val_dataset):,} examples")
    print(f"Max sequence length: {max_length} tokens")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Total training steps: {len(train_loader) * num_epochs}")
    print()
    
    # Test first batch
    first_batch = next(iter(train_loader))
    print(f"First batch shapes:")
    print(f"  Chosen input_ids: {first_batch['chosen_input_ids'].shape}")
    print(f"  Rejected input_ids: {first_batch['rejected_input_ids'].shape}")
    print()
    
    # Test model forward pass
    print("Testing model forward pass...")
    model.eval()
    with torch.no_grad():
        test_chosen_ids = first_batch['chosen_input_ids'].to(device)
        test_chosen_mask = first_batch['chosen_attention_mask'].to(device)
        test_rewards = model(test_chosen_ids, test_chosen_mask)
        print(f"Test rewards shape: {test_rewards.shape}")
        print(f"Test rewards: {test_rewards}")
    model.train()
    print()
    
    # Optimizer (for all parameters in RewardModel)
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    print("Starting preference training...")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in train_pbar:
            # Move batch to device
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            
            # Get rewards for chosen and rejected responses using RewardModel
            chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = model(rejected_input_ids, rejected_attention_mask)
            
            # Bradley-Terry loss: -log(sigmoid(chosen_reward - rejected_reward))
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at step {global_step}")
                print(f"Loss value: {loss.item()}")
                print(f"Chosen rewards: {chosen_rewards}")
                print(f"Rejected rewards: {rejected_rewards}")
                continue  # Skip this batch
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "reward_diff": f"{(chosen_rewards - rejected_rewards).mean().item():.4f}"
            })
            
            # Log to wandb
            wandb.log({
                "train_loss": loss.item(),
                "chosen_rewards_mean": chosen_rewards.mean().item(),
                "rejected_rewards_mean": rejected_rewards.mean().item(),
                "reward_diff_mean": (chosen_rewards - rejected_rewards).mean().item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "global_step": global_step
            })
            
            # Save checkpoint
            if global_step % save_steps == 0:
                model.save_reward_model(f"{output_dir}/checkpoint-{global_step}")
                wandb.log({"checkpoint_saved": global_step})
            
            # Evaluation
            if global_step % eval_steps == 0:
                val_loss = evaluate_preference(model, val_loader, device)
                if not torch.isnan(torch.tensor(val_loss)):
                    wandb.log({
                        "val_loss": val_loss,
                        "global_step": global_step
                    })
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.save_reward_model(f"{output_dir}/best-model")
                        wandb.log({"best_val_loss": val_loss})
                
                model.train()
        
        # Epoch summary
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")
        
        # Final evaluation for epoch
        val_loss = evaluate_preference(model, val_loader, device)
        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "epoch_train_loss": avg_train_loss,
            "epoch_val_loss": val_loss
        })
    
    # Save final model
    model.save_reward_model(f"{output_dir}/final-model")
    print(f"Training completed! Model saved to {output_dir}")
    
    # Close wandb
    wandb.finish()


def evaluate_preference(model, val_loader, device):
    """
    Evaluate model on validation set with Bradley-Terry loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            
            # Get rewards for chosen and rejected responses using RewardModel
            chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = model(rejected_input_ids, rejected_attention_mask)
            
            # Bradley-Terry loss
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train RewardModel with Bradley-Terry preference loss")
    parser.add_argument("--sft_model_path", default="./lora-output/best-model", help="Path to trained SFT model")
    parser.add_argument("--model_name", default="Qwen/Qwen1.5-0.5B", help="Model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--output_dir", default="./reward-output", help="Output directory")
    parser.add_argument("--wandb_project", default="qwen-reward-preference", help="Wandb project name")
    parser.add_argument("--wandb_run_name", default=None, help="Wandb run name")
    parser.add_argument("--dataset_name", default="CarperAI/openai_summarize_comparisons", help="Dataset name")
    
    args = parser.parse_args()
    
    train_preference(
        sft_model_path=args.sft_model_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        dataset_name=args.dataset_name
    )


if __name__ == "__main__":
    main() 