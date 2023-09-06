from flask import Flask, jsonify, request, render_template
import subprocess

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']

    try:
        completed_process = subprocess.run(
            ["docker", "run", "-e", f'PROMPT={question}', "zack-llama2:0.0.1"],
            capture_output=True, text=True
        )

        output = completed_process.stdout
        return jsonify({"response": output})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
