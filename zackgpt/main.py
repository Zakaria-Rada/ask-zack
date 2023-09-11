import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set the absolute path to the saved model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the current directory's absolute path
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "results", "llama-2-7b-miniguanaco")

# Load saved model and tokenizer from the local directory only once
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_SAVE_PATH)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)

# Create a text generation pipeline only once
GENERATOR = pipeline("text-generation", model=MODEL, tokenizer=TOKENIZER)


def generate_text(prompt):
    """Generate text using the pre-loaded model."""
    # Generate text using the provided prompt
    output = GENERATOR(prompt, max_length=100, num_return_sequences=1)
    generated_text = output[0]['generated_text']

    return generated_text


if __name__ == "__main__":
    while True:  # To continuously get user prompts and generate text until the user decides to stop
        # Provide a prompt for text generation
        user_prompt = input("\nEnter your prompt (or type 'exit' to quit): ")
        if user_prompt.strip().lower() == "exit":
            break

        response = generate_text(user_prompt)
        print("\nGenerated Text:\n", response)
