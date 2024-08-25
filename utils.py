def format_dataset(dataset, keys, instruction_col_name, response_col_name):
    """Format the dataset by retaining only necessary columns and renaming them."""
    cols_to_remove = [key for key in keys if key not in [instruction_col_name, response_col_name]]
    dataset = dataset.remove_columns(cols_to_remove)
    dataset = dataset.rename_column(instruction_col_name, "instruction")
    dataset = dataset.rename_column(response_col_name, "response")
    return dataset

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
