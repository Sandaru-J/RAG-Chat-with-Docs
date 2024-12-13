"""
    Author: Sandaru Jaythilaka
    Date: 11/22/24
    Description: simple implementaton for rag doc from end to development
                good for simple scnearios
    Efficency: good 
    Status: working fine
"""

import os
from typing import List
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file

class RAGDocumentQuerySystem:
    """
    Retrieval-Augmented Generation (RAG) system for querying documents.
    This class encapsulates document ingestion, embedding storage, retrieval, and answer generation.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the RAG system with OpenAI API key and text splitting parameters.
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")
        self.vectorstore = None
        self.llm = OpenAI(openai_api_key=openai_api_key)
        print("RAG System Initialized.")
    
    def load_and_split_documents(self, file_paths: List[str]) -> List[str]:
        """
        Load and split documents into smaller chunks.
        
        Args:
            file_paths (List[str]): List of file paths to load.
        
        Returns:
            List of document chunks.
        """
        print("Loading and splitting documents...")
        all_chunks = []
        for path in file_paths:
            loader = PyPDFLoader(path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
        print(f"Loaded and split {len(all_chunks)} chunks from {len(file_paths)} documents.")
        print(f'relasing chunks {all_chunks}')
        return all_chunks
    
    def create_vectorstore(self, document_chunks: List[str]):
        """
        Create a vectorstore with embeddings for the given document chunks.
        
        Args:
            document_chunks (List[str]): List of document chunks to store.
        """
        print("Creating vectorstore...")
        self.vectorstore = Chroma.from_documents(document_chunks, self.embeddings)
        print("Vectorstore created and ready for similarity search.")
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve top-k relevant documents based on the query.
        
        Args:
            query (str): The user's query.
            top_k (int): Number of top documents to retrieve.
        
        Returns:
            List of relevant document chunks.
        """
        print(f"Retrieving top {top_k} documents for query: {query}")
        if not self.vectorstore:
            raise ValueError("Vectorstore is not initialized. Please create the vectorstore first.")
        
        results = self.vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """
        Generate an answer based on the query and retrieved context.
        
        Args:
            query (str): The user's query.
            context (List[str]): Retrieved document chunks.
        
        Returns:
            str: Generated answer.
        """
        print("Generating answer using retrieved context...")
        combined_context = " ".join(context)
        prompt = (
            f"Using the following context, answer the query:\n\n"
            f"Context: {combined_context}\n\nQuery: {query}"
        )
        response = self.llm(prompt)
        return response
    
    def query_documents(self, query: str, top_k: int = 5) -> str:
        """
        Complete pipeline: retrieve documents and generate an answer.
        
        Args:
            query (str): The user's query.
            top_k (int): Number of top documents to retrieve.
        
        Returns:
            str: Final generated answer.
        """
        relevant_docs = self.retrieve_documents(query, top_k=top_k)
        return self.generate_answer(query, relevant_docs)


# Main function to demonstrate the RAG system
if __name__ == "__main__":
    # Initialize the RAG system
    rag_system = RAGDocumentQuerySystem()
    
    # Open file dialog to upload documents
    print("Please select the documents to upload:")
    Tk().withdraw()  # Hide the root window
    document_paths = askopenfilenames(
        title="Select Documents",
        filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
    )
    
    if not document_paths:
        print("No documents uploaded. Exiting.")
        exit(0)
    
    # Load, split, and index documents
    document_chunks = rag_system.load_and_split_documents(document_paths)
    print(f'these are the chunks {document_chunks}')
    rag_system.create_vectorstore(document_chunks)
    
    # Allow user to iteratively ask questions
    print("\nYou can now start asking questions about the uploaded documents.")
    while True:
        user_query = input("Enter your question (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Exiting. Goodbye!")
            break
        try:
            answer = rag_system.query_documents(user_query, top_k=5)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")

