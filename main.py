from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import subprocess
import os

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    os.chdir('/llama.cpp')
    command = ["./main", "-t", "8", "-ngl", "1", "-m", "llama-2-13b-chat.ggmlv3.q4_0.bin", "--color", "-c", "2048", "--temp", "0.7", "--repeat_penalty", "1.1", "-n", "-1", "-p", "[PROMPT] What is mean the name : ZAKARIA.? [/PROMPT]"]
    result = subprocess.run(command, capture_output=True, text=True)

    output = result.stdout
    html_content = f"<html><body><pre>{output}</pre></body></html>"
    return html_content
