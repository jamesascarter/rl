import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset
from functools import partial
from tqdm import tqdm
from models import RewardModel

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Fix tokenizer warnings

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

MODEL_NAME = "Qwen/Qwen1.5-0.5B"
DATASET_NAME = "CarperAI/openai_summarize_tldr"

SFT_ADAPTER_PATH = "./lora-output/best-model"
REWARD_ADAPTER_PATH = "./reward-output/best-model"
POLICY_OUTPUT_PATH = "./ppo-output" 

MAX_LEN = 256
EPOCHS = 1
BATCH_SIZE = 4

PPO_EPOCHS = 4
CLIP_EPS = 0.2

EVAL_INTERVAL = 100
SAVE_INTERVAL = 400


# --------------------------
# LOAD MODELS & TOKENIZER
# --------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # do the padding on the left side for generate such as [PAD PAD prompt prompt prompt]

# 1. Load the base model and prepare it for k-bit training
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
base_model = prepare_model_for_kbit_training(base_model)

# 2. Load the first adapter (SFT) to create the main PeftModel
# This is now our primary model object. The first adapter is automatically named 'sft'.
policy_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, adapter_name="sft")
print("Initialized PeftModel with SFT adapter as 'sft'.")

# 3. Load the additional adapters into the existing PeftModel
# Load the reward adapter and name it 'reward'

# Load the SFT weights again, but this time give them the new name 'policy'
# This creates our trainable policy adapter, initialized from the SFT checkpoint.
policy_model.load_adapter(SFT_ADAPTER_PATH, adapter_name="policy")
print("Loaded SFT weights as new adapter 'policy'.")

# Move the final multi-adapter model to the device
policy_model.to(DEVICE)

# 4. Define and instantiate the RewardModel wrapper
# It will use the SAME underlying model but with a specific head
reward_model = RewardModel(sft_model_path=SFT_ADAPTER_PATH, model_name=MODEL_NAME)
reward_model.load_reward_model(REWARD_ADAPTER_PATH)
reward_model.to(DEVICE)
reward_model.eval()

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, policy_model.parameters()), lr=1e-5)

# Update the optimizer to target only the 'policy' adapter's parameters

# Load dataset
dataset = load_dataset(DATASET_NAME, split="train")

# Filter + preprocess in one go
def preprocess_function(ex):
    if not ex["prompt"].strip().endswith("\nTL;DR:"):
        return {"keep": False}

    prompt = "Summarize:\n" + ex["prompt"].strip() + " "
    encoding = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LEN,
    )
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "keep": True
    }

dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)
dataset = dataset.filter(lambda ex: ex["keep"])
dataset = dataset.with_format("torch")

def collate_fn(batch):
    input_ids = [ex["input_ids"] for ex in batch]
    attention_masks = [ex["attention_mask"] for ex in batch]

    # Left pad: pad_sequence defaults to right, so reverse → pad → reverse again
    input_ids = [torch.flip(x, dims=[0]) for x in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    input_ids = torch.flip(input_ids, dims=[1])

    attention_masks = [torch.flip(x, dims=[0]) for x in attention_masks]
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    attention_masks = torch.flip(attention_masks, dims=[1])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks
    }

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True  # this helps CPU→GPU transfer
)

global_step = 0
running_loss = 0.0

# Better advantage computation
def compute_advantages(rewards, gamma=0.99, gae_lambda=0.95):
    """Compute GAE advantages"""
    advantages = []
    gae = 0
    
    for reward in reversed(rewards):
        gae = reward + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages, device=rewards.device)

for epoch in range(EPOCHS):
    loop = tqdm(loader)
    for batch in loop:
        try:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # === 1) Rollout ===
            # Set the active adapter to 'policy' for generation
            policy_model.set_adapter("policy")
            policy_model.train() # Use train mode for generation to ensure dropout is active if it was during SFT

            with torch.no_grad():
                output = policy_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_LEN,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Detach completions
            completions = []
            for j in range(BATCH_SIZE):
                gen = output[j][input_ids.shape[1]:]
                completions.append(gen.unsqueeze(0))
            completions = torch.cat(completions, dim=0)

            generated = [tokenizer.decode(completions[i], skip_special_tokens=True) for i in range(BATCH_SIZE)]

            # Combine prompt + generated
            full_ids = torch.cat([input_ids, completions], dim=1)
            full_attention_mask = torch.ones_like(full_ids).to(DEVICE)

            # === 2) Reward ===
            # Use the separate reward model (no adapter switching needed)
            with torch.no_grad():
                reward = reward_model(full_ids, full_attention_mask).detach()

            # Normalize rewards for stability
            reward = (reward - reward.mean()) / (reward.std() + 1e-8)

            # === 3) Compute old log probs ===
            # Set the active adapter to 'sft' (our fixed reference)
            policy_model.set_adapter("sft")
            with torch.no_grad():
                old_outputs = policy_model(
                    input_ids=full_ids,
                    attention_mask=full_attention_mask
                )
                old_logits = old_outputs.logits[:, :-1, :]
                old_labels = full_ids[:, 1:]

                old_logprobs_per_token = -nn.functional.cross_entropy(
                    old_logits.reshape(-1, old_logits.size(-1)),
                    old_labels.reshape(-1),
                    reduction='none'
                )
                old_logprobs = old_logprobs_per_token.view(old_labels.shape).sum(dim=1)

            # Add a value head to your model
            class ValueHead(nn.Module):
                def __init__(self, hidden_size):
                    super().__init__()
                    self.value_head = nn.Linear(hidden_size, 1)
                
                def forward(self, hidden_states):
                    return self.value_head(hidden_states).squeeze(-1)

            # Compute value estimates for better advantage
            value_head = ValueHead(policy_model.model.config.hidden_size).to(DEVICE)
            with torch.no_grad():
                last_hidden_state = policy_model.model.get_input_embeddings()(input_ids)
                value_estimates = value_head(last_hidden_state)

            advantage = reward - value_estimates

            # === 4) PPO update ===
            # Set the adapter back to 'policy' for the training step
            policy_model.set_adapter("policy")
            for ppo_epoch in range(PPO_EPOCHS):
                with autocast(DEVICE, dtype=torch.bfloat16):
                    outputs = policy_model(
                        input_ids=full_ids,
                        attention_mask=full_attention_mask
                    )
                    logits = outputs.logits[:, :-1, :]
                    labels = full_ids[:, 1:]

                    log_probs = torch.log_softmax(logits, dim=-1)
                    log_probs_for_labels = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                    logprobs = log_probs_for_labels.sum(dim=1)

                    ratio = torch.exp(logprobs - old_logprobs.detach())
                    clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                    
                    # Normalize advantages
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    
                    surrogate1 = ratio * advantage
                    surrogate2 = clipped_ratio * advantage
                    
                    ppo_loss = -torch.min(surrogate1, surrogate2).mean()
                    
                    # Add KL divergence monitoring
                    kl_div = (old_logprobs - logprobs).mean()
                    
                    # Early stopping if KL is too high
                    if kl_div > 0.1:  # target_kl
                        print(f"KL divergence {kl_div:.4f} > 0.1, stopping PPO updates")
                        break

                # The backward pass will now correctly only affect 'policy' adapter weights
                optimizer.zero_grad()
                ppo_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += ppo_loss.item()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️ OOM at global_step={global_step} batch={input_ids.shape}")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            else:
                raise

        global_step += 1

        if global_step % EVAL_INTERVAL == 0:
            avg_loss = running_loss / max(1, EVAL_INTERVAL)
            running_loss = 0.0
            
            prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

            print(f"=== Step {global_step} ===")
            print(f"**Prompt**: {prompt_text[:200]}...")
            print(f"**Generated**: {generated[0][:200]}...")
            print(f"**Reward**: {reward[0].item():.4f}")
            print(f"**Advantage**: {advantage[0].item():.4f}")
            print(f"**KL Div**: {kl_div.item():.4f}")
            print(f"**Avg PPO loss**: {avg_loss:.4f}")
            print(f"**Ratio**: {ratio[0].item():.4f}")

        # Save LoRA checkpoint
        if global_step % SAVE_INTERVAL == 0:
            ckpt_path = f"{POLICY_OUTPUT_PATH}_ckpt_step_{global_step}"
            policy_model.save_pretrained(ckpt_path)
            print(f"Saved interim LoRA to {ckpt_path}")

# Save only LoRA weights
policy_model.save_pretrained(POLICY_OUTPUT_PATH)
print(f"Saved policy LoRA adapter to {POLICY_OUTPUT_PATH}")