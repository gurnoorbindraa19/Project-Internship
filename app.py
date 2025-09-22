from pathlib import Path
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from gemini_wrapper import GeminiEmbeddings, GeminiChat
from langchain.prompts import PromptTemplate

import re
import shutil

app = Flask(__name__)

faiss_folder = Path("faiss_index")
docs_folder = Path(r"C:\hcl project\hcl-intellibot\documents")


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


rag_template = """
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

rag_prompt = PromptTemplate(input_variables=["context", "question"], template=rag_template)

# RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=GeminiChat(model_name="gemini-2.5-flash", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": rag_prompt}
)

# General LLM chain for fallback
general_llm_chain = LLMChain(
    llm=GeminiChat(model_name="gemini-2.5-flash", temperature=0),
    prompt=PromptTemplate(
        input_variables=["question"],
        template="You are a helpful assistant. Answer the following question briefly:\n\nQuestion: {question}\nAnswer:"
    )
)


def format_answer(text):
    if not text:
        return ""
    text = text.replace('\r', '').strip()
    text = re.sub(r'([A-Za-z\s,&-]+)(\d+\.)', r'\1\n\2', text)
    text = re.sub(r'(\d+\.)', r'\n\1', text)
    text = re.sub(r'\n(\d+\.)', r'\n   \1', text)
    text = re.sub(r'\n([A-Z][a-zA-Z]+)$', r'\n\n\1', text, flags=re.MULTILINE)
    url_pattern = r'(https?://[^\s]+)'
    text = re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def get_combined_answer(query):
    raw_answer = qa_chain.invoke({"query": query})["result"].strip()
    if not raw_answer or "I don't know" in raw_answer.lower() or len(raw_answer) < 10 or "Sorry" in raw_answer:
        raw_answer = general_llm_chain.run({"question": query}).strip()
    return format_answer(raw_answer)


@app.route("/get_answer", methods=["POST"])
def get_answer():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"answer": "⚠️ No query received."})
    formatted_answer = get_combined_answer(query)
    return jsonify({"answer": formatted_answer})

if __name__ == "__main__":
    app.run(debug=True)
