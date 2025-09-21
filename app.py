from pathlib import Path
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from gemini_wrapper import GeminiEmbeddings, GeminiChat
from langchain.prompts import PromptTemplate

import re
import shutil

# -------------------------
# Initialize Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Load FAISS + Documents
# -------------------------
faiss_folder = Path("faiss_index")
docs_folder = Path(r"C:\hcl project\hcl-intellibot\documents")

# Delete old index folder if exists
if faiss_folder.exists():
    shutil.rmtree(faiss_folder)

def load_documents():
    docs = []
    for file_path in docs_folder.glob("*.*"):
        if file_path.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(file_path)).load())
        elif file_path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(file_path)).load())
    return docs

if not faiss_folder.exists():
    print("🔹 Creating FAISS index from documents...")
    docs = load_documents()
    if not docs:
        print("⚠️ No documents found in docs/ folder.")
    else:
        db = FAISS.from_documents(docs, GeminiEmbeddings())
        db.save_local(faiss_folder)
        print("✅ FAISS index created and saved!")
else:
    print("🔹 Loading FAISS vector store...")
    db = FAISS.load_local(faiss_folder, GeminiEmbeddings(), allow_dangerous_deserialization=True)
    print("✅ FAISS vector store loaded!")

retriever = db.as_retriever()

# -------------------------
# Define Prompt + QA Chain
# -------------------------
template = """
You are HCL IntelliBot, a helpful assistant for employees of the company.

- If the answer is in the provided documents, use that info.
- If not in the docs but reasonable, answer briefly using general knowledge.
- If inappropriate/unrelated, respond: "Sorry, this is not in my knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

qa_chain = RetrievalQA.from_chain_type(
    llm=GeminiChat(model_name="gemini-2.5-flash", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

# -------------------------
# Formatter for structured answers with clickable links
# -------------------------
def format_answer(text):
    """
    Formats raw QA text into structured paragraphs with:
    - Headings separated
    - Numbered lists indented
    - URLs converted to clickable HTML links
    """
    if not text:
        return ""

    text = text.replace('\r', '').strip()

    # Add newline after headings if immediately followed by number
    text = re.sub(r'([A-Za-z\s,&-]+)(\d+\.)', r'\1\n\2', text)

    # Add newline before numbered list items (1., 2., 3.)
    text = re.sub(r'(\d+\.)', r'\n\1', text)

    # Indent numbered list items
    text = re.sub(r'\n(\d+\.)', r'\n   \1', text)

    # Extra line breaks for headings like HR, Legal, IT, Infrastructure
    text = re.sub(r'\n([A-Z][a-zA-Z]+)$', r'\n\n\1', text, flags=re.MULTILINE)

    # Convert URLs to clickable HTML links
    url_pattern = r'(https?://[^\s]+)'
    text = re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text)

    # Collapse multiple blank lines into two
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

# -------------------------
# Flask route for frontend
# -------------------------
@app.route("/get_answer", methods=["POST"])
def get_answer():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"answer": "⚠️ No query received."})

    # Run QA chain
    raw_answer = qa_chain.invoke({"query": query})["result"]

    # Format answer for clean display
    formatted_answer = format_answer(raw_answer)

    return jsonify({"answer": formatted_answer})

# -------------------------
# Run Flask app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
