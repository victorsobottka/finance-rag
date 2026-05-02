# app.py  — Gradio UI for Hugging Face Spaces
import gradio as gr
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from rag_chain import build_rag_chain

# Load pre-built vectorstore (commit chroma_db/ to your HF repo)
def load_chain():
    embeddings = OpenAIEmbeddings(
        api_key=os.environ["OPENAI_API_KEY"]  # set in HF Spaces secrets
    )
    vs = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="finance_docs"
    )
    return build_rag_chain(vs)

chain = load_chain()

EXAMPLE_QUESTIONS = [
    "What was gross margin last quarter?",
    "How did revenue compare to guidance?",
    "What risks did management highlight?",
    "What is the current cash position?",
]

def answer(question: str, history: list) -> str:
    if not question.strip():
        return "Please enter a question about the financial documents."
    try:
        return chain.invoke(question)
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.ChatInterface(
    fn=answer,
    title="Finance RAG — Earnings Report Q&A",
    description="Ask questions about uploaded earnings reports and SEC filings. Answers cite specific passages.",
    examples=EXAMPLE_QUESTIONS,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()

# --- Deployment steps ---
# 1. huggingface-cli login
# 2. Create new Space: https://huggingface.co/new-space
# 3. Select Gradio SDK
# 4. Add OPENAI_API_KEY in Space Settings > Secrets
# 5. git push (include chroma_db/ folder with pre-built index)
