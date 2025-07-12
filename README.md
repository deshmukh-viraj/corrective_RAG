# 🏛️ Legal RAG System with Self-Correction

An AI-powered Legal Document Analysis System that uses **LangGraph**, **LangChain**, **Groq LLM**, and **Gradio** to process and analyze legal documents (PDF/DOCX/TXT). It performs retrieval-augmented generation (RAG) with automated self-correction, document verification, and citation of legal clauses.

## 🚀 Features

- ✅ **Upload legal documents** (PDF, DOCX, TXT)
- 🔍 **Ask legal questions** in natural language
- 🔄 **Self-correcting answer generation** (multi-iteration LangGraph)
- 📋 **Cites specific document clauses** and legal terms
- 🔒 **Verifies answer accuracy**, hallucinations, and missing information
- 📊 **Confidence scores**, correction logs, and source tracking
- 🎯 **Easy-to-use Gradio UI**

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Groq API key

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/legal-rag-system.git
   cd legal-rag-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file and add your API keys
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

## 🔧 Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=legal-rag-system
```

## 📖 Usage

1. **Launch the application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:7860`

3. **Upload legal documents**
   - Click "Upload Documents" and select your PDF, DOCX, or TXT files
   - Wait for processing and indexing to complete

4. **Ask legal questions**
   - Type your question in natural language
   - The system will analyze documents and provide citations
   - Review confidence scores and correction logs

## 🏗️ Architecture

### Core Components

- **Document Processing**: Extracts and chunks legal documents
- **Vector Store**: Stores document embeddings for retrieval
- **LangGraph Workflow**: Manages self-correction iterations
- **Groq LLM**: Provides language understanding and generation
- **Gradio Interface**: User-friendly web interface

### Self-Correction Pipeline

1. **Initial Answer Generation**: Creates first response from retrieved documents
2. **Verification Step**: Checks for accuracy and completeness
3. **Correction Iteration**: Refines answer based on verification
4. **Citation Addition**: Adds specific document references
5. **Confidence Scoring**: Provides reliability metrics

## 🔍 API Reference

### Main Functions

#### `process_documents(files)`
Processes uploaded legal documents and creates vector embeddings.

**Parameters:**
- `files` (List[File]): List of uploaded document files

**Returns:**
- `status` (str): Processing status message

#### `ask_legal_question(question, chat_history)`
Processes legal questions using RAG with self-correction.

**Parameters:**
- `question` (str): Legal question in natural language
- `chat_history` (List): Previous conversation history

**Returns:**
- `answer` (str): Generated answer with citations
- `confidence` (float): Confidence score (0-1)
- `sources` (List[str]): Referenced document sources



## 🙏 Acknowledgments

- LangChain team for the excellent framework
- Groq for providing fast LLM inference
- Gradio for the intuitive UI framework
- The open-source community for various dependencies

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/legal-rag-system/issues) page
2. Create a new issue with detailed description
3. Join our [Discord community](https://discord.gg/your-discord) for discussions

## 🔄 Changelog

### v1.0.0 (2024-01-15)
- Initial release with basic RAG functionality
- Self-correction pipeline implementation
- Gradio web interface
- Support for PDF, DOCX, and TXT documents

---

**⚠️ Disclaimer**: This system is for informational purposes only and should not be considered as legal advice. Always consult with qualified legal professionals for legal matters.
