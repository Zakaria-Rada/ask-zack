from flask import Flask, render_template, request, jsonify
import subprocess
import json

app = Flask(__name__)


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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run', methods=['POST'])
def run():
    prompt = request.json['prompt']
    result = run_docker_command(prompt)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6060)
