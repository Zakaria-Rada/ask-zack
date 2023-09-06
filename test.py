import subprocess
import json


def run_docker_command(prompt):
    try:
        completed_process = subprocess.run(
            ["docker", "run", "-e", f'PROMPT={prompt}', "zack-llama2:0.0.1"],
            capture_output=True,
            text=True
        )

        output = completed_process.stdout
        return {"response": output}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    prompt = input("Please enter your prompt: ")
    result = run_docker_command(prompt)
    print(json.dumps(result, indent=4))
