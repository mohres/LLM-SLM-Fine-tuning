import argparse

import torch
from hydra import compose, initialize
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig


def get_config(config_name: str):
    with initialize(config_path="./configs", version_base="1.1"):
        cfg = compose(config_name=config_name)

    return cfg


def format_dataset(dataset, keys, instruction_col_name, response_col_name):
    """Format the dataset by retaining only necessary columns and renaming them."""
    cols_to_remove = [
        key for key in keys if key not in [instruction_col_name, response_col_name]
    ]
    dataset = dataset.remove_columns(cols_to_remove)
    dataset = dataset.rename_column(instruction_col_name, "instruction")
    dataset = dataset.rename_column(response_col_name, "response")
    return dataset


def prepare_datasets(dataset, instruction_col_name, response_col_name):
    """Format and split the dataset for training and evaluation."""
    available_cols = list(dataset["train"].features.keys())
    formatted_dataset = format_dataset(
        dataset, available_cols, instruction_col_name, response_col_name
    )

    if "valid" in formatted_dataset:
        train_dataset = formatted_dataset["train"]
        eval_dataset = formatted_dataset["valid"]
    elif "test" in formatted_dataset:
        train_dataset = formatted_dataset["train"]
        eval_dataset = formatted_dataset["test"]
    else:
        split_dataset = formatted_dataset["train"].train_test_split(test_size=0.2)
        train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]

    return train_dataset, eval_dataset


def generate_response(model, tokenizer, instruction, device="cpu"):
    """Generate a response from the model based on an instruction."""
    messages = [{"role": "user", "content": instruction}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs, max_new_tokens=128, temperature=0.2, top_p=0.9, do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def parse_arguments():
    """Parse command line arguments."""
    model_options = ["Mistral", "smolLM"]
    parser = argparse.ArgumentParser(description="Choose a model for training.")
    parser.add_argument(
        "--model",
        type=str,
        choices=model_options,
        default="smolLM",
        help=f"Specify the model (options: {', '.join(model_options)}, default: smolLM)",
    )
    return parser.parse_args()


def setup_output_directory(train_cfg, model_cfg, dataset_cfg):
    """Determine the directory where fine-tuned models and logs are saved.

    Returns:
        str: The path to the output directory.
    """
    if train_cfg.output_dir:
        return train_cfg.output_dir

    if model_cfg.quantization:
        return f"{model_cfg.id.split('/')[-1]}-{dataset_cfg.id.split('/')[-1]}-Instruct-{model_cfg.quantization}Q"
    return f"{model_cfg.id.split('/')[-1]}-{dataset_cfg.id.split('/')[-1]}-Instruct"


def load_tokenizer(model_cfg):
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.id, use_fast=model_cfg.use_fast_tokenizer
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def setup_device_and_quantization(model_cfg):
    """Set up the device and quantization configuration."""
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    quantization_config = None
    if use_cuda and model_cfg.quantization:
        quantization_mapping = {
            4: BitsAndBytesConfig(load_in_4bit=True),
            8: BitsAndBytesConfig(load_in_8bit=True),
        }
        quantization_config = quantization_mapping.get(model_cfg.quantization)
        if quantization_config is None:
            raise ValueError(
                f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}"
            )

    return device, quantization_config


def load_model(model_cfg, quantization_config, device):
    """Load and configure the model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if quantization_config is None:
        model.to(device)
    return model


def create_peft_config(model_cfg):
    """Create and return the PEFT configuration."""
    return LoraConfig(
        r=model_cfg.lora.peft_r,
        lora_alpha=model_cfg.lora.peft_alpha,
        lora_dropout=model_cfg.lora.peft_dropout,
        bias=model_cfg.lora.peft_bias,
        target_modules=model_cfg.lora.target_modules,
    )


def create_sft_config(train_cfg, output_dir):
    """Create and return the SFT configuration."""
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.epochs,
        max_seq_length=train_cfg.max_seq_length,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        optim=train_cfg.optim,
        save_steps=train_cfg.save_steps,
        logging_steps=train_cfg.logging_steps,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        fp16=train_cfg.fp16,
        bf16=train_cfg.bf16,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        packing=train_cfg.packing,
    )


def formatting_prompts_func(example: dict, tokenizer) -> str:
    """Format chat prompt for training."""
    chat = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    return text


def print_example(example):
    """Print an example from the dataset."""
    print(f"Original Dataset Example:")
    print(f"Instruction: {example['instruction']}")
    print(f"Response: {example['response']}")
    print("-" * 100)


def print_response(response):
    """Print the model's response."""
    print(f"Model response:")
    print(response.split("assistant\n")[-1])
    print("-" * 100)
