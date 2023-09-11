
# Install required packages
# !pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

# Configuration parameters
# Model related
model_name = "NousResearch/llama-2-7b-chat-hf"  # Original model
dataset_name = "mlabonne/guanaco-llama2-1k"     # Training dataset
new_model = "llama-2-7b-miniguanaco"            # Name for the fine-tuned model

# QLoRA specific parameters
lora_r = 64          # LoRA attention dimension
lora_alpha = 16      # Alpha parameter for LoRA scaling
lora_dropout = 0.1   # Dropout probability for LoRA layers

# BitsAndBytes (quantization) parameters
use_4bit = True
bnb_4bit_compute_dtype = "float16"  # Compute data type for 4-bit models
bnb_4bit_quant_type = "nf4"        # Quantization type (fp4 or nf4)
use_nested_quant = False           # Nested quantization for 4-bit models

# Training related parameters
output_dir = "./results"
num_train_epochs = 1
per_device_train_batch_size = 4
gradient_accumulation_steps = 1
optim = "paged_adamw_32bit"
learning_rate = 2e-4
weight_decay = 0.001
max_grad_norm = 0.3
max_steps = -1                  # Number of training steps (overrides num_train_epochs)
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 25
fp16 = False
bf16 = False
max_seq_length = None
packing = False                # Pack multiple short examples in the same input sequence
device_map = {"": 0}

# Load dataset
dataset = load_dataset(dataset_name, split="train")

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

# Define training arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# Train model using SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing
)

trainer.train()
trainer.model.save_pretrained(new_model)
