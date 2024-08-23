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
    print(response)
    print("-" * 100)
