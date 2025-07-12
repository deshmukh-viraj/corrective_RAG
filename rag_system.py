import json
import logging
from typing import Dict, Literal, Any, List
from dataclasses import dataclass

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START,END,StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from config import Config
from document_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGState(TypedDict):
    question: str
    context: List[str]
    answer: str
    confidence: float
    sources: List[str]
    corrections_made: List[str]
    verification_status: str
    iteration_count: int
    metadata: Dict[str, Any]

@dataclass
class RAGResponse:
    answer: str
    confidence: float
    sources: List[str]
    corrections_made: List[str]
    verification_status: str
    metadata: Dict[str, Any]

class LegalRAGSystem:
    def __init__(self):
        self.llm = ChatGroq(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE
        )
        self.document_processor = DocumentProcessor()
        self.memory = MemorySaver()
        self.graph = self.create_correction_graph()

        self.answer_prompt = ChatPromptTemplate.from_template("""
        You are an expert legal analyst. Use the provided context to answer the question accurately and comprehensively.

        Context from legal documents:
        {context}
        Question: {question}                                                   

        Instructions:
        1. Provide a precise, fact-based answer using ONLY the information from the contect
        2. Include specific contract clauses, sections or legal terms when relevant
        3. if information is not in the context state "This information not provided in document"
        4. cite specific sections or clauses when making claims
        5. be through out concise in your analysis
        6. focus on legal accuracy and implications
        Answer:
        """)

        self.verification_prompt = ChatPromptTemplate.from_template("""
        You are a legal verification expert. Analyze the following answer for accuracy, completeness, and legal precision.
        
        Original Question: {question}
        Generated Answer: {answer}
        Source Context: {context}
        
        Verification Tasks:
        1. Check if ALL claims in the answer are supported by the context
        2. Identify any potential hallucinations or unsupported statements
        3. Verify that legal terms and concepts are used correctly
        4. Assess if the answer fully addresses the question
        5. Check for any missing critical information available in the context
        
        Rate the answer on a scale of 0-1 for confidence based on:
        - Factual accuracy (context support)
        - Completeness (addresses all aspects)
        - Legal precision (correct terminology)
        - Relevance (answers the specific question)
        
        Respond in JSON format:
        {{
            "verification_status": "VERIFIED" or "NEEDS_CORRECTION",
            "confidence": 0.0-1.0,
            "issues": ["specific issues found"],
            "missing_information": ["important information that should be included"],
            "legal_concerns": ["any legal accuracy concerns"]
        }}
        """)
        
        self.correction_prompt = ChatPromptTemplate.from_template("""
        You are tasked with correcting a legal analysis response based on verification feedback.
        
        Original Question: {question}
        Previous Answer: {answer}
        Context: {context}
        Issues Identified: {issues}
        
        Correction Instructions:
        1. Address each identified issue specifically
        2. Ensure every claim is supported by the context
        3. Include any missing critical information from the context
        4. Maintain legal accuracy and precision
        5. If certain information is truly not available in the context, state this clearly
        6. Provide citations to specific parts of the context when possible
        
        Provide the corrected answer:
        """)

    def create_correction_graph(self) -> StateGraph:
        workflow = StateGraph(RAGState)

        workflow.add_node("retrieve", self.retrieve_context)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("verify", self.verify_answer)
        workflow.add_node("correct", self.correct_answer)
        workflow.add_node("finalize", self.finalize_response)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve","generate")
        workflow.add_edge("generate","verify")
        workflow.add_conditional_edges(
            "verify",
            self.should_correct,
            {
                "correct":"correct",
                "finalize":"finalize"
            }
        )
        workflow.add_edge("correct", "verify")
        workflow.add_edge("finalize", END)

        return workflow.compile(checkpointer=self.memory)
    
    def retrieve_context(self, state: RAGState) -> RAGState:
        """Retrieve relevant context from vectorstore"""
        try:
            docs = self.document_processor.search_documents(state["question"])
            context = [doc.page_content for doc in docs]
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            
            # Extract metadata
            metadata = {
                "retrieved_docs": len(docs),
                "source_files": list(set(sources)),
                "chunk_sizes": [len(doc.page_content) for doc in docs]
            }
            
            return {
                **state,
                "context": context,
                "sources": sources,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            raise
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using retrieved context"""
        try:
            context_str = "\n\n".join(state["context"])
            
            response = self.llm.invoke(
                self.answer_prompt.format(
                    context=context_str,
                    question=state["question"]
                )
            )
            
            return {
                **state,
                "answer": response.content,
                "iteration_count": state.get("iteration_count", 0) + 1
            }
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    def verify_answer(self, state: RAGState) -> RAGState:
        """Verify the generated answer"""
        try:
            context_str = "\n\n".join(state["context"])
            
            verification_response = self.llm.invoke(
                self.verification_prompt.format(
                    question=state["question"],
                    answer=state["answer"],
                    context=context_str
                )
            )
            
            # Parse verification response
            try:
                verification_data = json.loads(verification_response.content)
                
                # Compile all issues
                all_issues = []
                all_issues.extend(verification_data.get("issues", []))
                all_issues.extend(verification_data.get("missing_information", []))
                all_issues.extend(verification_data.get("legal_concerns", []))
                
                return {
                    **state,
                    "verification_status": verification_data["verification_status"],
                    "confidence": verification_data["confidence"],
                    "corrections_made": state.get("corrections_made", []) + all_issues
                }
            except json.JSONDecodeError:
                logger.warning("Failed to parse verification response as JSON")
                return {
                    **state,
                    "verification_status": "NEEDS_CORRECTION",
                    "confidence": 0.5,
                    "corrections_made": state.get("corrections_made", []) + ["Verification parsing error"]
                }
        except Exception as e:
            logger.error(f"Error verifying answer: {str(e)}")
            raise
    
    def correct_answer(self, state: RAGState) -> RAGState:
        """Correct the answer based on verification feedback"""
        try:
            context_str = "\n\n".join(state["context"])
            issues_str = "\n".join(state.get("corrections_made", []))
            
            correction_response = self.llm.invoke(
                self.correction_prompt.format(
                    question=state["question"],
                    answer=state["answer"],
                    context=context_str,
                    issues=issues_str
                )
            )
            
            return {
                **state,
                "answer": correction_response.content,
                "corrections_made": state.get("corrections_made", []) + ["Answer corrected based on verification"]
            }
        except Exception as e:
            logger.error(f"Error correcting answer: {str(e)}")
            raise
    
    def finalize_response(self, state: RAGState) -> RAGState:
        """Finalize the response"""
        # Add final metadata
        final_metadata = state.get("metadata", {})
        final_metadata.update({
            "final_iteration_count": state.get("iteration_count", 0),
            "total_corrections": len(state.get("corrections_made", []))
        })
        
        return {
            **state,
            "metadata": final_metadata
        }
    
    def should_correct(self, state: RAGState) -> str:
        """Determine if correction is needed"""
        # Don't correct if we've exceeded max iterations
        if state.get("iteration_count", 0) >= Config.MAX_CORRECTION_ITERATIONS:
            logger.info(f"Max iterations reached: {Config.MAX_CORRECTION_ITERATIONS}")
            return "finalize"
        
        # Correct if verification failed or confidence is low
        if (state.get("verification_status") == "NEEDS_CORRECTION" or 
            state.get("confidence", 0) < Config.MIN_CONFIDENCE_THRESHOLD):
            logger.info(f"Correction needed. Status: {state.get('verification_status')}, Confidence: {state.get('confidence')}")
            return "correct"
        
        return "finalize"
    
    def process_documents(self, file_paths: List[str]) -> Dict:
        """Process multiple document files"""
        results = []
        total_chunks = 0
        
        for file_path in file_paths:
            result = self.document_processor.process_uploaded_file(file_path)
            results.append(result)
            if result["success"]:
                total_chunks += result["chunks_added"]
        
        return {
            "results": results,
            "total_chunks_added": total_chunks,
            "vectorstore_stats": self.document_processor.get_vectorstore_stats()
        }
    
    async def query(self, question: str, session_id: str = "default") -> RAGResponse:
        """Query the RAG system with self-correction"""
        try:
            # Initialize state
            initial_state = {
                "question": question,
                "context": [],
                "answer": "",
                "confidence": 0.0,
                "sources": [],
                "corrections_made": [],
                "verification_status": "",
                "iteration_count": 0,
                "metadata": {}
            }
            
            # Run the graph
            config = {"configurable": {"thread_id": session_id}}
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            return RAGResponse(
                answer=final_state["answer"],
                confidence=final_state["confidence"],
                sources=final_state["sources"],
                corrections_made=final_state["corrections_made"],
                verification_status=final_state["verification_status"],
                metadata=final_state["metadata"]
            )
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            raise
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "document_stats": self.document_processor.get_vectorstore_stats(),
            "config": {
                "model": Config.LLM_MODEL,
                "chunk_size": Config.CHUNK_SIZE,
                "max_retrieval_docs": Config.MAX_RETRIEVAL_DOCS,
                "confidence_threshold": Config.MIN_CONFIDENCE_THRESHOLD,
                "max_iterations": Config.MAX_CORRECTION_ITERATIONS
            }
        }