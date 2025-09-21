from flask import Flask, render_template, request, redirect, url_for, jsonify
from app import qa_chain  # Import qa_chain from app.py
import re

app = Flask(__name__)

# --- Function to clean answers ---
def clean_answer(ans):
    """
    - Removes *, •, - bullets/stars
    - Keeps new lines
    - Strips extra spaces
    """
    ans = re.sub(r"[*•-]\s*", "", ans)  # remove stars or bullets
    ans = ans.strip()
    return ans

# Route 1 – Instructions page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('chat'))
    return render_template('index.html')

# Route 2 – Chat page
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    answer = None
    query = None
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            result = qa_chain.invoke({"query": query})
            answer_text = result.get("result", "").strip()
            answer_text = clean_answer(answer_text)

            if not answer_text or "I don't know" in answer_text.lower() or len(answer_text) < 10:
                answer_text = "Sorry, this is not in my knowledge base."

            answer = answer_text

    return render_template('chat.html', answer=answer, query=query)

# Route 3 – JSON API for quick buttons
@app.route('/get_answer', methods=['POST'])
def get_answer():
    query = request.form.get('query')
    if not query:
        return jsonify({"answer": "No query received."})

    result = qa_chain.invoke({"query": query})
    answer_text = result.get("result", "").strip()
    answer_text = clean_answer(answer_text)
    
    if not answer_text or "I don't know" in answer_text.lower() or len(answer_text) < 10:
        answer_text = "Sorry, this is not in my knowledge base."
    
    return jsonify({"answer": answer_text})

if __name__ == "__main__":
    app.run(debug=True)
