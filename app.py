from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    prompt = request.json['prompt']
    docker_command = f"docker run -e PROMPT='{prompt}' zack-llama2:0.0.1"

    try:
        result = subprocess.run(docker_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"An error occurred: {e}"

    return jsonify({'response': output})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
