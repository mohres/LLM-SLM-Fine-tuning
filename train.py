import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from utils import (
    format_dataset,
    generate_response,
    get_config,
    print_example,
    print_response,
)


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


def formatting_prompts_func(example: dict) -> str:
    """Format prompt for training."""
    text = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
    return text


def main():
    # Load environment variables
    cfg = get_config("train_smolLM")
    model_cfg = cfg.model
    dataset_cfg = cfg.dataset
    train_cfg = cfg.training_arguments
    
    # Setup logging
    output_dir = (
        train_cfg.output_dir
        if train_cfg.output_dir
        else f"{model_cfg.id.split('/')[-1]}-{dataset_cfg.id.split('/')[-1]}-{train_cfg.epochs}epochs"
    )
    logger.info(f"Model ID: {model_cfg.id}")
    logger.info(f"Dataset ID: {dataset_cfg.id}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Training Epochs: {train_cfg.epochs}")

    # Load dataset
    dataset = load_dataset(dataset_cfg.id)
    train_dataset, eval_dataset = prepare_datasets(
        dataset, dataset_cfg.instruction_column_name, dataset_cfg.response_column_name
    )

    logger.info(f"Training Dataset: {len(train_dataset)} examples")
    logger.info(f"Evaluation Dataset: {len(eval_dataset)} examples")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.id, use_fast=model_cfg.use_fast_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Generate a response for the first example in the validation dataset
    example1 = eval_dataset[0]
    response = generate_response(model, tokenizer, example1["instruction"], device)

    print_example(example1)
    print_response(response)

    # PEFT configuration
    peft_config = LoraConfig(
        r=model_cfg.lora.peft_r,
        lora_alpha=model_cfg.lora.peft_alpha,
        lora_dropout=model_cfg.lora.peft_dropout,
        bias=model_cfg.lora.peft_bias,
        target_modules=model_cfg.lora.target_modules,
    )

    # SFT configuration
    sft_config = SFTConfig(
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

    # Initialize trainer and start training
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        peft_config=peft_config,
        args=sft_config,
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model()
    if cfg.hugging_face.push_to_hub and (token := cfg.hugging_face.token):
        trainer.push_to_hub(token=token)

    # Load fine-tuned model and generate response
    ft_model = AutoModelForCausalLM.from_pretrained(output_dir).to(device)
    response = generate_response(ft_model, tokenizer, example1["instruction"], device)

    print_example(example1)
    print_response(response)


if __name__ == "__main__":
    main()
