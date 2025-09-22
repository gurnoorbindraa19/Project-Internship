from flask import Flask, render_template, request, redirect, url_for, jsonify
from app import get_combined_answer  # Use the new combined function
import re

app = Flask(__name__)


def clean_answer(ans):
    ans = re.sub(r"[*•-]\s*", "", ans)  # remove stars or bullets
    ans = ans.strip()
    return ans

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('chat'))
    return render_template('index.html')


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    answer = None
    query = None
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            answer_text = get_combined_answer(query)
            answer_text = clean_answer(answer_text)
            if not answer_text or len(answer_text) < 10:
                answer_text = "Sorry, this is not in my knowledge base."
            answer = answer_text
    return render_template('chat.html', answer=answer, query=query)


@app.route('/get_answer', methods=['POST'])
def get_answer():
    query = request.form.get('query')
    if not query:
        return jsonify({"answer": "No query received."})
    answer_text = get_combined_answer(query)
    answer_text = clean_answer(answer_text)
    if not answer_text or len(answer_text) < 10:
        answer_text = "Sorry, this is not in my knowledge base."
    return jsonify({"answer": answer_text})

if __name__ == "__main__":
    app.run(debug=True)
