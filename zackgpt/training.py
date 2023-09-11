"""
Fine-tuning script for QLoRA-based models using the Hugging Face Transformers library.
"""

import os
import torch
import gc
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


if __name__ == "__main__":

    # Configuration details
    model_name = "NousResearch/Llama-2-7b-chat-hf"
    dataset_name = "mlabonne/guanaco-llama2-1k"
    new_model = "llama-2-7b-miniguanaco"

    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1

    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False

    output_dir = "./results"
    num_train_epochs = 1
    fp16 = False
    bf16 = False
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 1
    gradient_checkpointing = True
    max_grad_norm = 0.3
    learning_rate = 2e-4
    weight_decay = 0.001
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "cosine"
    max_steps = -1
    warmup_ratio = 0.03
    group_by_length = True
    save_steps = 0
    logging_steps = 25

    max_seq_length = None
    packing = False
    device_map = {"": 0}

    save_directory = os.path.join(output_dir, new_model)

    try:
        dataset = load_dataset(dataset_name, split="train")
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

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
            lr_scheduler_type=lr_scheduler_type,
            report_to="tensorboard"
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=packing,
        )

        trainer.train()

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        trainer.model.save_pretrained(save_directory)

    except Exception as e:
        print(f"An error occurred during training: {e}")

    try:
        logging.set_verbosity(logging.CRITICAL)
        prompt = "What is a large language model?"
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        print(result[0]['generated_text'])

    except Exception as e:
        print(f"An error occurred during text generation: {e}")

    del model
    del pipe
    del trainer
    gc.collect()
    gc.collect()

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(base_model, save_directory)
        model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)

        print(f"Model and tokenizer saved to {save_directory}")

    except Exception as e:
        print(f"An error occurred during model merging or saving: {e}")
