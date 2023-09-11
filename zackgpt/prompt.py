import os
import warnings
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Suppress all warnings
warnings.simplefilter("ignore")

torch.cuda.empty_cache()

GPU_DEVICE = 1 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "results", "llama-2-7b-miniguanaco")

# Load saved model and tokenizer from the local directory only once
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_SAVE_PATH)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)

# Move the model to the specified GPU
MODEL = MODEL.to(f'cuda:{GPU_DEVICE}')  

# Create a text generation pipeline only once
GENERATOR = pipeline("text-generation", model=MODEL, tokenizer=TOKENIZER, device=GPU_DEVICE)


def generate_text(prompt):
    """Generate text using the pre-loaded model."""
    # Generate text using the provided prompt
    output = GENERATOR(prompt, max_length=200, num_return_sequences=1)
    generated_text = output[0]['generated_text']

    generated_text = generated_text[len(prompt):]

    return generated_text


if __name__ == "__main__":
    print("Chatbot is ready. Type 'exit' to end the conversation.\n")
    while True:
        # User's message
        user_prompt = input("You: ")

        if user_prompt.strip().lower() == "exit":
            break

        # Chatbot's response
        response = generate_text(user_prompt)
        print("ZackGPT:", response)

        torch.cuda.empty_cache()
        gc.collect()
