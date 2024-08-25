import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from utils import format_dataset, generate_response, print_example, print_response


def load_environment_variables():
    """Load and return environment variables."""
    load_dotenv()
    env_vars = {
        "model_id": os.getenv("MODEL_ID"),
        "dataset_id": os.getenv("DATASET_ID"),
        "instruction_col_name": os.getenv("INSTRUCTION_COLUMN_NAME"),
        "response_col_name": os.getenv("RESPONSE_COLUMN_NAME"),
        "peft_r": int(os.getenv("PEFT_R")),
        "peft_lora_alpha": int(os.getenv("PEFT_LORA_ALPHA")),
        "peft_lora_dropout": float(os.getenv("PEFT_LORA_DROPOUT")),
        "peft_bias": os.getenv("PEFT_BIAS"),
        "sft_num_train_epochs": int(os.getenv("SFT_NUM_TRAIN_EPOCHS")),
        "sft_max_seq_length": int(os.getenv("SFT_MAX_SEQ_LENGTH")),
        "sft_per_device_train_batch_size": int(
            os.getenv("SFT_PER_DEVICE_TRAIN_BATCH_SIZE")
        ),
        "sft_gradient_accumulation_steps": int(
            os.getenv("SFT_GRADIENT_ACCUMULATION_STEPS")
        ),
        "sft_gradient_checkpointing": os.getenv("SFT_GRADIENT_CHECKPOINTING").lower()
        in ["true", "1", "t", "y", "yes"],
        "sft_optim": os.getenv("SFT_OPTIM"),
        "sft_save_steps": int(os.getenv("SFT_SAVE_STEPS")),
        "sft_logging_steps": int(os.getenv("SFT_LOGGING_STEPS")),
        "sft_learning_rate": float(os.getenv("SFT_LEARNING_RATE")),
        "sft_weight_decay": float(os.getenv("SFT_WEIGHT_DECAY")),
        "sft_fp16": os.getenv("SFT_FP16").lower() in ["true", "1", "t", "y", "yes"],
        "sft_bf16": os.getenv("SFT_BF16").lower() in ["true", "1", "t", "y", "yes"],
        "sft_warmup_ratio": float(os.getenv("SFT_WARMUP_RATIO")),
        "sft_lr_scheduler_type": os.getenv("SFT_LR_SCHEDULER_TYPE"),
        "packing": os.getenv("PACKING").lower() in ["true", "1", "t", "y", "yes"],
        "hf_access_token": os.getenv("HF_ACCESS_TOKEN"),
    }
    return env_vars


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
    env_vars = load_environment_variables()

    # Setup logging
    output_dir = f"{env_vars['model_id'].split('/')[-1]}-{env_vars['dataset_id'].split('/')[-1]}-{env_vars['sft_num_train_epochs']}epochs"
    logger.info(f"Model ID: {env_vars['model_id']}")
    logger.info(f"Dataset ID: {env_vars['dataset_id']}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Training Epochs: {env_vars['sft_num_train_epochs']}")

    # Load dataset
    dataset = load_dataset(env_vars["dataset_id"])
    train_dataset, eval_dataset = prepare_datasets(
        dataset, env_vars["instruction_col_name"], env_vars["response_col_name"]
    )

    logger.info(f"Training Dataset: {len(train_dataset)} examples")
    logger.info(f"Evaluation Dataset: {len(eval_dataset)} examples")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(env_vars["model_id"])
    model = AutoModelForCausalLM.from_pretrained(env_vars["model_id"]).to(device)

    # Generate a response for the first example in the training dataset
    example1 = eval_dataset[0]
    response = generate_response(model, tokenizer, example1["instruction"], device)

    print_example(example1)
    print_response(response)

    # Data collator for training
    response_template = "assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # PEFT configuration
    peft_config = LoraConfig(
        r=env_vars["peft_r"],
        lora_alpha=env_vars["peft_lora_alpha"],
        lora_dropout=env_vars["peft_lora_dropout"],
        bias=env_vars["peft_bias"],
    )

    # SFT configuration
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=env_vars["sft_num_train_epochs"],
        max_seq_length=env_vars["sft_max_seq_length"],
        per_device_train_batch_size=env_vars["sft_per_device_train_batch_size"],
        gradient_accumulation_steps=env_vars["sft_gradient_accumulation_steps"],
        gradient_checkpointing=env_vars["sft_gradient_checkpointing"],
        optim=env_vars["sft_optim"],
        save_steps=env_vars["sft_save_steps"],
        logging_steps=env_vars["sft_logging_steps"],
        learning_rate=env_vars["sft_learning_rate"],
        weight_decay=env_vars["sft_weight_decay"],
        fp16=env_vars["sft_fp16"],
        bf16=env_vars["sft_bf16"],
        warmup_ratio=env_vars["sft_warmup_ratio"],
        lr_scheduler_type=env_vars["sft_lr_scheduler_type"],
        packing=env_vars["packing"],
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
    if token := env_vars["hf_access_token"]:
        trainer.push_to_hub(token=token)

    # Load fine-tuned model and generate response
    ft_model = AutoModelForCausalLM.from_pretrained(output_dir).to(device)
    response = generate_response(ft_model, tokenizer, example1["instruction"], device)

    print_example(example1)
    print_response(response)


if __name__ == "__main__":
    main()
