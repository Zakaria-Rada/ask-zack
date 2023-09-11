import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def generate_text(prompt):
    # Set the absolute path to the saved model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the current directory's absolute path
    model_save_path = os.path.join(BASE_DIR, "results", "llama-2-7b-miniguanaco")

    # Load saved model and tokenizer from the local directory
    loaded_model = AutoModelForCausalLM.from_pretrained(model_save_path)
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_save_path)

    # Create a text generation pipeline
    generator = pipeline("text-generation", model=loaded_model, tokenizer=loaded_tokenizer)

    # Generate text using the provided prompt
    output = generator(prompt, max_length=100, num_return_sequences=1)
    generated_text = output[0]['generated_text']

    return generated_text


if __name__ == "__main__":
    # Provide a prompt for text generation
    user_prompt = input("Enter your prompt: ")
    response = generate_text(user_prompt)
    print("\nGenerated Text:\n", response)
