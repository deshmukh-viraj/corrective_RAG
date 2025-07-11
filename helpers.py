import os
import shutil
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_uploaded_file(uploaded_file, upload_dir: str) -> str:
    """Save uploaded file to specified directory"""
    try:
        upload_path = Path(upload_dir)
        upload_path.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename if file already exists
        file_path = upload_path / uploaded_file.name
        counter = 1
        while file_path.exists():
            name_parts = uploaded_file.name.rsplit('.', 1)
            if len(name_parts) == 2:
                name, ext = name_parts
                file_path = upload_path / f"{name}_{counter}.{ext}"
            else:
                file_path = upload_path / f"{uploaded_file.name}_{counter}"
            counter += 1
        
        # Save file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)
        
        logger.info(f"Saved uploaded file: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise

def format_confidence_score(confidence: float) -> str:
    """Format confidence score as percentage with color coding"""
    percentage = int(confidence * 100)
    
    if percentage >= 90:
        color = "🟢"
    elif percentage >= 70:
        color = "🟡"
    else:
        color = "🔴"
    
    return f"{color} {percentage}%"

def format_verification_status(status: str) -> str:
    """Format verification status with appropriate emoji"""
    if status == "VERIFIED":
        return "✅ VERIFIED"
    elif status == "NEEDS_CORRECTION":
        return "⚠️ NEEDS CORRECTION"
    else:
        return "❓ UNKNOWN"

def format_corrections_list(corrections: List[str]) -> str:
    """Format corrections list for display"""
    if not corrections:
        return "No corrections needed ✅"
    
    formatted_corrections = []
    for i, correction in enumerate(corrections, 1):
        formatted_corrections.append(f"{i}. {correction}")
    
    return "\n".join(formatted_corrections)

def format_sources_list(sources: List[str]) -> str:
    """Format sources list for display"""
    if not sources:
        return "No sources available"
    
    # Remove duplicates while preserving order
    unique_sources = []
    seen = set()
    for source in sources:
        if source not in seen:
            unique_sources.append(source)
            seen.add(source)
    
    formatted_sources = []
    for i, source in enumerate(unique_sources, 1):
        formatted_sources.append(f"{i}. 📄 {source}")
    
    return "\n".join(formatted_sources)

def format_metadata(metadata: Dict[str, Any]) -> str:
    """Format metadata for display"""
    if not metadata:
        return "No metadata available"
    
    formatted_lines = []
    
    # Document statistics
    if "retrieved_docs" in metadata:
        formatted_lines.append(f"📊 Retrieved Documents: {metadata['retrieved_docs']}")
    
    if "source_files" in metadata:
        formatted_lines.append(f"📁 Source Files: {len(metadata['source_files'])}")
    
    if "final_iteration_count" in metadata:
        formatted_lines.append(f"🔄 Processing Iterations: {metadata['final_iteration_count']}")
    
    if "total_corrections" in metadata:
        formatted_lines.append(f"🔧 Total Corrections: {metadata['total_corrections']}")
    
    return "\n".join(formatted_lines) if formatted_lines else "No metadata available"

def validate_file_type(file_path: str, allowed_extensions: List[str]) -> bool:
    """Validate if file type is allowed"""
    file_extension = Path(file_path).suffix.lower()
    return file_extension in allowed_extensions

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    return Path(file_path).stat().st_size / (1024 * 1024)

def format_file_info(file_path: str) -> str:
    """Format file information for display"""
    path = Path(file_path)
    size_mb = get_file_size_mb(file_path)
    
    return f"📄 {path.name} ({size_mb:.1f} MB)"

def create_gradio_examples() -> List[List[str]]:
    """Create example questions for Gradio interface"""
    return [
        ["What are the key termination clauses in this contract?"],
        ["What are the payment terms and conditions?"],
        ["Are there any liability limitations or indemnification clauses?"],
        ["What are the governing law and jurisdiction provisions?"],
        ["What are the confidentiality and non-disclosure requirements?"],
        ["Are there any penalty or liquidated damages clauses?"],
        ["What are the intellectual property rights and ownership terms?"],
        ["What are the force majeure provisions?"],
        ["Are there any renewal or extension options?"],
        ["What are the dispute resolution mechanisms?"]
    ]

def create_processing_summary(results: Dict) -> str:
    """Create a summary of document processing results"""
    if not results or "results" not in results:
        return "No processing results available"
    
    successful = sum(1 for r in results["results"] if r["success"])
    total = len(results["results"])
    total_chunks = results.get("total_chunks_added", 0)
    
    summary_lines = [
        f"✅ Successfully processed: {successful} / {total}" 
    ]