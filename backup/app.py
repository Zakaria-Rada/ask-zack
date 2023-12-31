from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json

app = Flask(__name__)
CORS(app)


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


@app.route('/run', methods=['POST'])
def run_command():
    data = request.json
    prompt = data.get("prompt")
    result = run_docker_command(prompt)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=True)
