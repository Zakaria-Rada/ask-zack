# run_llama.py
import subprocess

def run_llama(prompt):
    model = "llama-2-13b-chat.ggmlv3.q4_0.bin"
    cmd_components = [
        "./main",
        "-t 8",
        "-ngl 1",
        f"-m {model}",
        "--color",
        "-c 2048",
        "--temp 0.7",
        "--repeat_penalty 1.1",
        "-n -1",
        f"-p '[PROMPT] {prompt} [/PROMPT]'"
    ]
    cmd = " ".join(cmd_components)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout
