import gradio as gr
import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from config import Config
from rag_system import LegalRAGSystem
from helpers import (
    save_uploaded_file,
    format_confidence_score,
    format_verification_status,
    format_corrections_list,
    format_sources_list,
    format_metadata,
    create_gradio_examples,
    create_processing_summary,
    validate_file_type,
    format_file_info
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalRAGApp:
    """Main application class for the Legal RAG System"""
    
    def __init__(self):
        # Validate configuration
        Config.validate_config()
        
        # Initialize RAG system
        self.rag_system = LegalRAGSystem()
        
        # Track uploaded files
        self.uploaded_files = []
        
    def process_files(self, files: List[gr.File]) -> tuple:
        """Process uploaded files and return status"""
        if not files:
            return "❌ No files uploaded", "", ""
        
        try:
            # Save uploaded files
            file_paths = []
            file_info_lines = []
            
            for file in files:
                # Validate file type
                if not validate_file_type(file.name, Config.LEGAL_DOCUMENT_TYPES):
                    return f"❌ Unsupported file type: {Path(file.name).suffix}", "", ""
                
                # Save file
                file_path = save_uploaded_file(file, str(Config.UPLOADS_DIR))
                file_paths.append(file_path)
                file_info_lines.append(format_file_info(file_path))
                
                # Track uploaded file
                if file_path not in self.uploaded_files:
                    self.uploaded_files.append(file_path)
            
            # Process documents
            results = self.rag_system.process_documents(file_paths)
            
            # Create summary
            summary = create_processing_summary(results)
            file_info = "\n".join(file_info_lines)
            
            # Get system stats
            stats = self.rag_system.get_system_stats()
            stats_info = format_metadata(stats["document_stats"])
            
            return summary, file_info, stats_info
            
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            return f"❌ Error processing files: {str(e)}", "", ""
    
    def query_documents(self, question: str, session_id: str = "default") -> tuple:
        """Query the RAG system and return formatted response"""
        if not question.strip():
            return "❓ Please enter a question", "", "", "", "", ""
        
        if not self.uploaded_files:
            return "📁 Please upload legal documents first", "", "", "", "", ""
        
        try:
            # Run async query
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                self.rag_system.query(question, session_id)
            )
            loop.close()
            
            # Format response
            answer = response.answer
            confidence = format_confidence_score(response.confidence)
            status = format_verification_status(response.verification_status)
            corrections = format_corrections_list(response.corrections_made)
            sources = format_sources_list(response.sources)
            metadata = format_metadata(response.metadata)
            
            return answer, confidence, status, corrections, sources, metadata
            
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            return f"❌ Error: {str(e)}", "", "", "", "", ""
    
    def clear_documents(self) -> tuple:
        """Clear all uploaded documents"""
        try:
            # Reset RAG system
            self.rag_system = LegalRAGSystem()
            self.uploaded_files = []
            
            return "✅ All documents cleared", "", ""
            
        except Exception as e:
            logger.error(f"Error clearing documents: {str(e)}")
            return f"❌ Error clearing documents: {str(e)}", "", ""
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        with gr.Blocks(
            title="Legal RAG System",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
            .main-header {
                text-align: center;
                margin-bottom: 2rem;
            }
            .status-box {
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
            }
            """
        ) as interface:
            
            # Header
            gr.Markdown(
                """
                # 🏛️ Legal RAG System
                ### AI-Powered Legal Document Analysis with Self-Correction
                Upload legal documents and get precise, verified answers to your questions.
                """,
                elem_classes=["main-header"]
            )
            
            with gr.Row():
                # Left column - Document Management
                with gr.Column(scale=1):
                    gr.Markdown("## 📁 Document Management")
                    
                    # File upload
                    file_upload = gr.File(
                        label="Upload Legal Documents",
                        file_count="multiple",
                        file_types=Config.LEGAL_DOCUMENT_TYPES,
                        height=120
                    )
                    
                    # Processing buttons
                    with gr.Row():
                        process_btn = gr.Button(
                            "📄 Process Documents",
                            variant="primary"
                        )
                        clear_btn = gr.Button(
                            "🗑️ Clear All",
                            variant="secondary"
                        )
                    
                    # Status displays
                    processing_status = gr.Textbox(
                        label="Processing Status",
                        interactive=False,
                        elem_classes=["status-box"]
                    )
                    
                    file_info = gr.Textbox(
                        label="Uploaded Files",
                        interactive=False,
                        lines=3
                    )
                    
                    system_stats = gr.Textbox(
                        label="System Statistics",
                        interactive=False,
                        lines=3
                    )
                
                # Right column - Q&A Interface
                with gr.Column(scale=2):
                    gr.Markdown("## 🤖 Legal Q&A")
                    
                    # Question input
                    question_input = gr.Textbox(
                        label="Ask a Question",
                        placeholder="What are the key termination clauses in this contract?",
                        lines=2
                    )
                    
                    # Submit button
                    submit_btn = gr.Button(
                        "🔍 Get Answer",
                        variant="primary",
                        size="lg"
                    )
                    
                    # Response section
                    with gr.Column():
                        answer_output = gr.Textbox(
                            label="📝 Answer",
                            interactive=False,
                            lines=8,
                            elem_classes=["status-box"]
                        )
                        
                        # Metadata row
                        with gr.Row():
                            confidence_output = gr.Textbox(
                                label="🎯 Confidence",
                                interactive=False,
                                scale=1
                            )
                            
                            verification_output = gr.Textbox(
                                label="✅ Verification",
                                interactive=False,
                                scale=1
                            )
                        
                        # Additional info in accordion
                        with gr.Accordion("📊 Detailed Information", open=False):
                            corrections_output = gr.Textbox(
                                label="🔧 Corrections Made",
                                interactive=False,
                                lines=3
                            )
                            
                            sources_output = gr.Textbox(
                                label="📚 Sources",
                                interactive=False,
                                lines=3
                            )
                            
                            metadata_output = gr.Textbox(
                                label="📈 Processing Metadata",
                                interactive=False,
                                lines=3
                            )
            
            # Example questions
            gr.Markdown("## 💡 Example Questions")
            examples = gr.Examples(
                examples=create_gradio_examples(),
                inputs=[question_input],
                label="Click on an example to use it:"
            )
            
            # Footer
            gr.Markdown(
                """
                ---
                **Note:** This system uses AI to analyze legal documents. Always verify important information with qualified legal professionals.
                """,
                elem_classes=["footer"]
            )
            
            # Event handlers
            process_btn.click(
                fn=self.process_files,
                inputs=[file_upload],
                outputs=[processing_status, file_info, system_stats]
            )
            
            clear_btn.click(
                fn=self.clear_documents,
                outputs=[processing_status, file_info, system_stats]
            )
            
            submit_btn.click(
                fn=self.query_documents,
                inputs=[question_input],
                outputs=[
                    answer_output,
                    confidence_output,
                    verification_output,
                    corrections_output,
                    sources_output,
                    metadata_output
                ]
            )
            
            # Allow Enter key to submit
            question_input.submit(
                fn=self.query_documents,
                inputs=[question_input],
                outputs=[
                    answer_output,
                    confidence_output,
                    verification_output,
                    corrections_output,
                    sources_output,
                    metadata_output
                ]
            )
        
        return interface
    
    def run(self):
        """Run the application"""
        try:
            interface = self.create_interface()
            
            interface.launch(
                share=Config.GRADIO_SHARE,
                server_name=Config.GRADIO_SERVER_NAME,
                server_port=Config.GRADIO_SERVER_PORT,
                show_error=True,
                favicon_path=None
            )
            
        except Exception as e:
            logger.error(f"Error running application: {str(e)}")
            raise

def main():
    """Main entry point"""
    try:
        app = LegalRAGApp()
        app.run()
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()