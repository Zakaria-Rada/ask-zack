import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def generate_text(prompt):
    # Set the path to the saved model
    output_dir = "./results"
    new_model = "llama-2-7b-miniguanaco"
    model_save_path = os.path.join(output_dir, new_model)

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
