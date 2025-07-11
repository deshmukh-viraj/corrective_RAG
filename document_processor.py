import os
import hashlib
from typing import List, Dict, Optional
from pathlib import Path
import logging

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, processing and vectorization"""
    def __init__(self):
        self.embeddings=HuggingFaceEmbeddings(model=Config.EMBEDDING_MODEL)
        self.text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            seperators=["\n\n", "\n", ".","1","?",","," ",""]

        )
        self.vectorstore=None
        self.processed_doc={}

    def get_file_hash(self, file_path:str):
        """Generate hash for file"""
        hash_md5=hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in filter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def load_documents(self, file_path:str) -> List[Document]:
        """Load docs based on its file type"""
        file_path=Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        file_size_mb=file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > Config.MAX_FILE_SIZE_MB:
            raise ValueError(f"File is too large: {file_size_mb: .1f}MB. Maximum allowed: {Config.MAX_FILE_SIZE_MB}MB")
        file_extention=file_path.suffix.lower()

        try:
            if file_extention == ".pdf":
                loader=PyPDFLoader(str(file_path))
            elif file_extention == ".docx":
                loader=Docx2txtLoader(str(file_path))
            elif file_extention == ".txt":
                loader=TextLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_extention}")
            
            documents=loader.load()

            for doc in documents:
                doc.metadata.update({
                    "source":file_path.name,
                    "file_type":file_extention,
                    "file_size_mn": round(file_size_mb,2),
                    "file_hash": self.get_file_hash(str(file_path))
                })

            logger.info(f"Loaded{len(documents)} pages from {file_path.name}")
            return documents
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")



    def process_documents(self, documents: List[Document]) -> List[Document]:
        """plit docs into chunks"""
        try:
            chunks=self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
        except Exception as e:
            logger.error(f"Error Processing documents: {str(e)}")
            raise

    def create_vectorstore(self, chunks:List[Document]) ->FAISS:
        try:
            self.vectorstore=FAISS.from_documents(chunks, self.embeddings)
            logger.info(f"Created vectorstore with {len(chunks)} chunks")
            return self.vectorstore
        except Exception as e:
            logger.error(f"Error creating vectorestore: {str(e)}")
            raise

    def add_docs_to_vectorstore(self, chunks:List[Document]):
        if self.vectorstore is None:
            return self.create_vectorstore(chunks)
        
        try:
            self.vectorstore.add_documents(chunks)
            logger.info(f"Added {len(chunks)} chunks to existing vectorstore")
        except Exception as e:
            logger.error(f"Error adding documents to vectorstore: {str(e)}")
            raise

    def process_uploaded_file(self, file_path:str) -> Dict:
        try:
            file_hash=self.get_file_hash(file_path)
            if file_hash in self.processed_doc:
                logger.info(f"File already processed: {Path(file_path).name}")
                return{
                    "success":True,
                    "message":"File already processed",
                    "chunks_added": 0,
                    "total_chunks":len(self.processed_doc[file_hash])
                }
            documents=self.load_documents(file_path)
            chunks=self.process_documents(documents)

            self.add_docs_to_vectorstore(chunks)
            self.processed_doc[file_hash]=chunks

            return {
                "success":True,
                "message":"document processed successfully",
                "chunks_added": len(chunks),
                "total_chunks": len(chunks)
            }
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing file: {str(e)}",
                "chunks_added": 0,
                "total_chunks":0
            }
        
    def search_documents(self, query: str, k: int=None) -> List[Document]:
        if self.vectorstore is None:
            raise ValueError("No documents loaded. Please upload documents first")
        k = k or Config.MAX_RETRIEVAL_DOCS

        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Retrievd {len(docs)} documents for query: {query[:50]}")
            return docs
        except Exception as e:
            logger.error(f"Error searching documetns: {str(e)}")
            raise

    
    def get_vectorestore_stats(self) -> Dict:
        if self.vectorstore is None:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "processed_files": 0
            
            }
        total_chunks = len(self.vectorstore.docstore._dict)

        return {
            "total_documents": len(self.processed_doc),
            "total_chunks": total_chunks,
            "processed_files": len(self.processed_doc)
        }