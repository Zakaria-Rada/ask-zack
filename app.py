from flask import Flask, request, jsonify, render_template
import subprocess
import json

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/run', methods=['POST'])
def run_command():
    prompt = request.form.get('prompt')
    try:
        completed_process = subprocess.run(
            ["docker", "run", "-e", f'PROMPT={prompt}', "zack-llama2:0.0.1"],
            capture_output=True,
            text=True
        )
        output = completed_process.stdout
        return jsonify({"response": output})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
