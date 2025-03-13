from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Optional
import os

def setup_vector_store(documents: List[Document], 
                      persist_directory: str = "./chroma_db",
                      embedding_model: str = "text-embedding-ada-002") -> Chroma:
    """
    Set up a Chroma vector store with the provided documents.
    
    Args:
        documents: List of documents to store
        persist_directory: Directory to persist the vector store
        embedding_model: OpenAI embedding model to use
        
    Returns:
        Chroma vector store
    """
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model=embedding_model)
    
    # Check if vector store already exists
    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        # Load existing vector store
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Add new documents if provided
        if documents:
            vectorstore.add_documents(documents)
            vectorstore.persist()
    else:
        # Create new vector store if it doesn't exist
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    
    return vectorstore

def add_document_to_vectorstore(vectorstore: Chroma, 
                              document: List[Document], 
                              document_id: str) -> bool:
    """
    Add a document to the vector store with proper metadata.
    
    Args:
        vectorstore: The Chroma vector store
        document: List of document chunks to add
        document_id: Unique identifier for the document
        
    Returns:
        Success status
    """
    try:
        # Add document ID to metadata
        for doc in document:
            if 'metadata' not in doc or doc.metadata is None:
                doc.metadata = {}
            doc.metadata['document_id'] = document_id
        
        # Add to vector store
        vectorstore.add_documents(document)
        vectorstore.persist()
        return True
    except Exception as e:
        print(f"Error adding document to vector store: {str(e)}")
        return False

def delete_document_from_vectorstore(vectorstore: Chroma, document_id: str) -> bool:
    """
    Delete a document from the vector store by ID.
    
    Args:
        vectorstore: The Chroma vector store
        document_id: ID of the document to delete
        
    Returns:
        Success status
    """
    try:
        vectorstore.delete(filter={"document_id": document_id})
        vectorstore.persist()
        return True
    except Exception as e:
        print(f"Error deleting document from vector store: {str(e)}")
        return False