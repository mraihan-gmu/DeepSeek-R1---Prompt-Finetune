# DeepSeek-R1---Prompt-Finetune
A Python Notebook to prompt and finetune DeepSeek-R1

This repository contains code for finetuning DeepSeek-R1-Distill-Qwen-1.5B model using medical domain data with Chain-of-Thought prompting. The implementation uses the Unsloth library for efficient training and optimization.

## 🌟 Features

- 🔥 Efficient finetuning using Unsloth optimization
- 🏥 Medical domain-specific training
- 💭 Chain-of-Thought (CoT) prompting
- 🚀 LoRA for parameter-efficient training
- 📊 WandB integration for training monitoring

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.11+
- PyTorch 2.0+
- Transformers
- Unsloth
- Datasets
- Wandb
- Accelerate
- Peft
- TRL
- xFormers

## 💻 Hardware Requirements

- GPU with at least 16GB VRAM (Tested on A100 40GB)
- CUDA compatible system

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/deepseek-r1-medical-cot.git
cd deepseek-r1-medical-cot
```

2. **Install dependencies**
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
pip install triton xformers
```

3. **Run the notebook**
- Open `DeepSeek_R1_Finetuning.ipynb` in Jupyter or Google Colab
- Follow the step-by-step instructions in the notebook

## 📘 Training Process

The notebook contains the following key steps:

1. **Installing Libraries**: Setup of all required dependencies
2. **Loading Pretrained Model**: Initialization of DeepSeek-R1
3. **LoRA Configuration**: Setting up parameter-efficient finetuning
4. **Dataset Preparation**: Loading and formatting medical domain data
5. **Training**: Finetuning process with monitoring
6. **Model Saving**: Saving the finetuned model

## 🎛️ Key Parameters

```python
# Training Arguments
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 2e-4
max_steps = 60
warmup_steps = 5

# LoRA Config
r = 64
lora_alpha = 16
lora_dropout = 0
```

## 📊 Training Monitoring

The training process is monitored using Weights & Biases (WandB). You'll need to:
1. Create a WandB account
2. Login using your API key
3. Monitor training metrics in real-time

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.


## 🙏 Acknowledgments

- DeepSeek AI for the base model
- Unsloth team for optimization library
- Medical-O1-Reasoning-SFT dataset creators

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---
