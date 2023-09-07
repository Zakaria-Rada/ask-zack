from flask import Flask, request, jsonify
import subprocess
import json


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


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


app = Flask(__name__)

@app.route('/run', methods=['POST'])
def run():
    if not request.is_json:
        return jsonify({"error": "Expected JSON payload"}), 415

    prompt = request.json.get('prompt', None)
    result = run_docker_command(prompt)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6060)
