import os
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    """Custom exception for vector store operations."""
    pass

def load_and_process_data(file_path: str) -> List[Document]:
    """Load and process data from CSV file.
    
    Args:
        file_path: Path to the CSV file containing conversation data
        
    Returns:
        List of Document objects ready for vector store
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        VectorStoreError: If there's an error processing the data
    """
    try:
        logger.info(f"Loading data from {file_path}...")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")
        
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Process each conversation into a document
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Combine relevant columns into text
        for _, row in df.iterrows():
            # Create context from instruction (prompt) and input-output pair
            context = f"Context: {row['instruction']}"
            conversation = f"User: {row['input']}\nAssistant: {row['output']}"
            text = f"{context}\n\n{conversation}"
            
            # Split into chunks if needed
            chunks = text_splitter.split_text(text)
            
            # Create Document objects
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "chunk": i,
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
        
        logger.info(f"Processed {len(documents)} document chunks")
        return documents
        
    except Exception as e:
        raise VectorStoreError(f"Error processing data: {str(e)}")

def setup_vector_store(
    documents: List[Document], 
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: Optional[str] = "./chroma_db"
) -> Chroma:
    """
    Set up the vector store with documents.
    
    Args:
        documents: List of documents to store
        embedding_model: HuggingFaceEmbeddings embedding model to use
        persist_directory: Directory to persist the vector store to
        
    Returns:
        Chroma vector store
        
    Raises:
        VectorStoreError: If there's an error setting up the vector store
    """
    try:
        logger.info("Setting up vector store...")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        try:
            # Try to load existing store first
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            logger.info("Loaded existing vector store")
            
        except Exception as e:
            # If loading fails, create new store
            logger.info("Creating new vector store...")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            # No need to call persist() as it's handled by from_documents with persist_directory
            logger.info("Vector store setup complete at {}".format(persist_directory))
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error setting up vector store: {str(e)}")
        raise VectorStoreError(f"Vector store setup failed: {str(e)}")

def add_document_to_vectorstore(
    vectorstore: Chroma, 
    documents: Document, 
    document_id: str
) -> bool:
    """
    Add one or more documents to the vector store with proper metadata.
    
    Args:
        vectorstore: The Chroma vector store
        documents: Single document or list of document chunks to add
        document_id: Unique identifier for the document(s)
        
    Returns:
        bool: True if successful, False otherwise
        
    Raises:
        VectorStoreError: If there's an error adding the document
    """
    try:
        # Convert single document to list
        doc_list = [documents] if isinstance(documents, Document) else documents
        
        # Add document ID to metadata
        for doc in doc_list:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata['document_id'] = document_id
        
        vectorstore.add_documents(doc_list)
        vectorstore.persist()
        logger.info(f"Successfully added document {document_id}")
        return True
    except Exception as e:
        logger.error(f"Error adding document to vector store: {str(e)}")
        raise VectorStoreError(f"Failed to add document: {str(e)}")

def delete_document_from_vectorstore(vectorstore: Chroma, document_id: str) -> bool:
    """
    Delete a document from the vector store by ID.
    
    Args:
        vectorstore: The Chroma vector store
        document_id: ID of the document to delete
        
    Returns:
        bool: True if successful, False otherwise
        
    Raises:
        VectorStoreError: If there's an error deleting the document
    """
    try:
        vectorstore.delete(filter={"document_id": document_id})
        vectorstore.persist()
        logger.info(f"Successfully deleted document {document_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting document from vector store: {str(e)}")
        raise VectorStoreError(f"Failed to delete document: {str(e)}")