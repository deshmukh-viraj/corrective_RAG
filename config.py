import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    LLM_MODEL =" deepseek-r1-distill-llama-70b"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_TEMPERATURE = 0.0

    CHUNK_SIZE=1000
    CHUNK_OVERLAP=200
    MAX_RETRIEVAL_DOCS=5
    MIN_CONFIDENCE_THRESHOLD=0.7
    MAX_CORRECTION_ITERATIONS=3

    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    UPLOADS_DIR = DATA_DIR / "uploads"

    GRADIO_SHARE = False
    GRADIO_SERVER_NAME = "0.0.0.0"
    GRADIO_SERVER_PORT = 7860

    LEGAL_DOCUMENTS_TYPES = [".pdf", ".docx", ".txt"]
    MAX_FILE_SIZE_MB = 50

    @classmethod
    def setup_directories(cls):
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.UPLOADS_DIR.mkdir(exist_ok=True)

    @classmethod
    def validate_config(cls):
        if not cls.HUGGINGFACE_API_KEY:
            raise ValueError(f" Hugging Face key not configured")
        cls.setup_directories()
        return True