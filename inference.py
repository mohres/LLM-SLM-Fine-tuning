import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import print_response, generate_response


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "mohres/SmolLM-135M-Instruct-medical_meadow_medical_flashcards-10epochs"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    while True:
        instruction = input("Instruction: ")
        response = generate_response(model, tokenizer, instruction, device)
        print_response(response)


if __name__ == "__main__":
    main()
