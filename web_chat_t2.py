"""
    Author: Sandaru Jaythilaka
    Date: 11/22/24
    Description: simple implementaton with rag to chat with url web pages
    Efficency: good
    Status: working fine
"""

import os
import logging
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain_community.vectorstores import Chroma
import requests
from bs4 import BeautifulSoup

# Set up logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    """
    A class for scraping the content from a webpage.
    """
    def __init__(self, url: str):
        self.url = url
    
    def fetch_content(self) -> str:
        """
        Fetches the content of the webpage and extracts the text.
        """
        try:
            response = requests.get(self.url)
            response.raise_for_status()  # Will raise HTTPError for bad requests (4xx, 5xx)
            soup = BeautifulSoup(response.text, 'html.parser')
            logger.info(f"Successfully fetched content from {self.url}")
            return soup.get_text(separator=' ', strip=True)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching webpage: {e}")
            raise Exception(f"Failed to fetch the webpage: {e}")


class TextProcessor:
    """
    A class for processing the raw text into manageable chunks for embedding.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def split_text(self, text: str) -> List[str]:
        """
        Splits the raw webpage text into chunks.
        """
        try:
            chunks = self.splitter.split_text(text)
            logger.info(f"Text successfully split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            raise Exception(f"Error processing text into chunks: {e}")


class EmbeddingsStore:
    """
    A class for creating embeddings for text chunks and storing them in a vector store.
    """
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vector_store = None
    
    def create_embeddings(self, chunks: List[str]) -> Chroma:
        """
        Creates embeddings for the given text chunks and stores them in Chroma vector store.
        """
        try:
            self.vector_store = Chroma.from_texts(chunks, embedding=self.embeddings)
            logger.info("Embeddings created and stored in Chroma vector store.")
            return self.vector_store
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise Exception(f"Error creating embeddings: {e}")


class QueryHandler:
    """
    A class to handle querying the vector store for answers based on the provided question.
    """
    def __init__(self, openai_api_key: str):
        self.llm = OpenAI(api_key=openai_api_key)
    
    def query_rag(self, question: str, vector_store: Chroma) -> str:
        """
        Retrieves relevant chunks from the vector store based on the question and generates an answer.
        """
        try:
            docs = vector_store.similarity_search(question, k=3)
            context = " ".join([doc.page_content for doc in docs])
            prompt = f"Answer the following question based on the provided context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
            response = self.llm(prompt)  # Generate the answer using OpenAI model
            logger.info(f"Generated answer for question: {question}")
            return response
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            raise Exception(f"Error generating answer: {e}")


class RAGWebQuerySystem:
    """
    A high-level class that integrates all components and handles the full workflow.
    """
    def __init__(self, openai_api_key: str, url: str):
        """
        Initializes the RAG system components.
        """
        self.url = url
        self.web_scraper = WebScraper(url)
        self.text_processor = TextProcessor()
        self.embeddings_store = EmbeddingsStore(openai_api_key)
        self.query_handler = QueryHandler(openai_api_key)
    
    def run(self):
        """
        The main workflow to fetch the content, process it, create embeddings, and handle queries.
        """
        try:
            # Step 1: Fetch the webpage content
            logger.info("Starting the process...")
            content = self.web_scraper.fetch_content()
            
            # Step 2: Split the content into chunks
            chunks = self.text_processor.split_text(content)
            
            # Step 3: Create embeddings and store them
            vector_store = self.embeddings_store.create_embeddings(chunks)
            
            logger.info("The webpage content has been processed. You can now ask questions.")
            
            # Step 4: Handle user queries
            self.handle_queries(vector_store)
        
        except Exception as e:
            logger.error(f"Error during the RAG process: {e}")
            raise
    
    def handle_queries(self, vector_store: Chroma):
        """
        Interactively handles user queries and generates answers.
        """
        while True:
            question = input("Enter your question (or type 'exit' to quit): ")
            if question.lower() == "exit":
                logger.info("Exiting the application.")
                break
            try:
                # Get the answer from the RAG system
                answer = self.query_handler.query_rag(question, vector_store)
                print(f"Answer: {answer}")
            except Exception as e:
                logger.error(f"Error handling query: {e}")
                print(f"Error: {e}")


# Main entry point for the RAG system
if __name__ == "__main__":
    # Load the OpenAI API key from the .env file
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key is not set.")
        raise ValueError("OpenAI API key is not set.")
    
    # Input URL for the webpage to scrape and query
    url = input("Enter the URL of the webpage: ").strip()
    
    # Initialize the RAG system
    rag_system = RAGWebQuerySystem(openai_api_key=OPENAI_API_KEY, url=url)
    
    # Run the RAG system
    rag_system.run()
