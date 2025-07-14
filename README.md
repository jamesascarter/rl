# Qwen Fine-tuning on TL;DR Dataset

This project provides a complete pipeline for fine-tuning a small Qwen model on the TL;DR (Too Long; Didn't Read) dataset for text summarization tasks.

## Features

- **Small Model Support**: Uses Qwen1.5-0.5B (500M parameters) for efficient training
- **TL;DR Dataset**: Fine-tuned on the popular TL;DR dataset from HuggingFace
- **Easy-to-use API**: Simple interface for training and inference
- **Flexible Configuration**: Customizable training parameters
- **Memory Efficient**: Optimized for training on consumer hardware

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

## Quick Start

### 1. Test the Base Model

First, test the base model without fine-tuning:

```bash
python train.py --test_only
```

This will load the base Qwen model and generate a summary for a sample text.

### 2. Start Fine-tuning

Begin the fine-tuning process:

```bash
python train.py --num_epochs 3 --batch_size 4 --learning_rate 5e-5
```

### 3. Use the Fine-tuned Model

After training, test the fine-tuned model:

```bash
python train.py --load_model ./qwen-tldr-finetuned --test_only
```

## Usage Examples

### Basic Usage

```python
from models import QwenSftModel

# Initialize model
model = QwenSftModel()

# Generate summary
text = "Your long text here..."
summary = model.generate_summary(text)
print(summary)
```

### Custom Training

```python
from models import QwenSftModel

# Initialize with custom model
model = QwenSftModel(model_name="Qwen/Qwen1.5-1B")

# Custom training parameters
trainer = model.train(
    output_dir="./my-finetuned-model",
    num_epochs=5,
    batch_size=2,
    learning_rate=3e-5
)
```

### Load Fine-tuned Model

```python
from models import QwenSftModel

# Load your fine-tuned model
model = QwenSftModel()
model.load_finetuned_model("./qwen-tldr-finetuned")

# Use for inference
summary = model.generate_summary("Your text here...")
```

## Command Line Options

The `train.py` script supports various command-line arguments:

- `--model_name`: HuggingFace model name (default: "Qwen/Qwen1.5-0.5B")
- `--output_dir`: Directory to save the fine-tuned model (default: "./qwen-tldr-finetuned")
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 4)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--test_only`: Only test the model without training
- `--load_model`: Path to load a pre-trained model

## Model Architecture

The system uses:
- **Base Model**: Qwen1.5-0.5B (500M parameters)
- **Dataset**: TL;DR dataset from HuggingFace
- **Training**: Supervised fine-tuning with causal language modeling
- **Prompt Format**: "Summarize the following text:\n\n{text}\n\nSummary:"

## Hardware Requirements

### Minimum Requirements
- **RAM**: 8GB
- **GPU**: Not required (CPU training supported)
- **Storage**: 2GB free space

### Recommended Requirements
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 5GB free space

## Training Configuration

The default training configuration is optimized for consumer hardware:

- **Batch Size**: 4 (adjust based on your GPU memory)
- **Learning Rate**: 5e-5 (good starting point for fine-tuning)
- **Epochs**: 3 (sufficient for most summarization tasks)
- **Gradient Accumulation**: 4 (effective batch size = 16)

## Dataset Information

The TL;DR dataset contains:
- **Train**: ~300K examples
- **Validation**: ~30K examples
- **Format**: (text, summary) pairs
- **Source**: Reddit posts and their summaries

## Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce batch size: `--batch_size 2`
   - Use gradient accumulation: increase `gradient_accumulation_steps`

2. **Slow Training**:
   - Use GPU if available
   - Reduce dataset size for faster iteration

3. **Poor Quality Summaries**:
   - Increase training epochs
   - Adjust learning rate
   - Use larger base model

### Performance Tips

- Use mixed precision training (enabled by default on GPU)
- Monitor GPU memory usage during training
- Start with smaller dataset subset for testing

## Model Output

The fine-tuned model generates concise summaries in response to the prompt format:
```
Summarize the following text:

[Your input text]

Summary: [Generated summary]
```

## License

This project uses the Qwen model which is subject to its own license. Please refer to the HuggingFace model page for licensing information.

## Contributing

Feel free to submit issues and enhancement requests! 
