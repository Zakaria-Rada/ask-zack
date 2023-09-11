import os
import warnings
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

warnings.simplefilter("ignore")

GPU_DEVICE = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the current directory's absolute path
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "results", "llama-2-7b-miniguanaco")

# Load saved model and tokenizer from the local directory only once
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_SAVE_PATH)
MODEL = MODEL.to(f'cuda:{GPU_DEVICE}')  # Move the model to the specified GPU
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)

# Create a text generation pipeline only once
GENERATOR = pipeline("text-generation", model=MODEL, tokenizer=TOKENIZER, device=GPU_DEVICE)


def generate_text(prompt):
    """Generate text using the pre-loaded model."""
    # Generate text using the provided prompt
    output = GENERATOR(prompt, max_length=1000, num_return_sequences=1)
    generated_text = output[0]['generated_text']

    # Remove the initial user's prompt from the generated text
    generated_text = generated_text[len(prompt):]

    return generated_text


def print_real_time(text):
    """Prints the text character by character to simulate real-time typing."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(0.05)
    print()


if __name__ == "__main__":
    print("Chatbot is ready. Type 'exit' to end the conversation.\n")
    while True:
        # User's message
        user_prompt = input("You: ")

        if user_prompt.strip().lower() == "exit":
            break

        # Chatbot's response
        response = generate_text(user_prompt)
        print("ZackGPT: ", end='')
        print_real_time(response.strip())
