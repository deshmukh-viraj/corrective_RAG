# 🏛️ Legal RAG System with Self-Correction

This project is an AI-powered Legal Document Analysis System that uses **LangGraph**, **LangChain**, **Groq LLM**, and **Gradio** to process and analyze legal documents (PDF/DOCX/TXT). It performs **retrieval-augmented generation (RAG)** with **automated self-correction**, document verification, and citation of legal clauses.

---

## 🚀 Features

- ✅ Upload legal documents (PDF, DOCX, TXT)
- 🔍 Ask legal questions in natural language
- 🧠 Self-correcting answer generation (multi-iteration LangGraph)
- 🧾 Cites specific document clauses and legal terms
- 🔒 Verifies answer accuracy, hallucinations, and missing information
- 📈 Confidence scores, correction logs, and source tracking
- 🎨 Easy-to-use Gradio UI

---

## 🛠️ Tech Stack

| Component     | Description                               |
|---------------|-------------------------------------------|
| **LangGraph** | Stateful agentic flow (RAG + correction)  |
| **LangChain** | Prompting, retrieval, parsing             |
| **Groq API**  | Fast LLM inference (LLama3 / Mixtral)     |
| **Gradio**    | Frontend UI for file upload & Q&A         |
| **HuggingFace** | Embeddings for document indexing        |
| **SentenceTransformer** | all-MiniLM-L6-v2                |

---

## 📂 Folder Structure

