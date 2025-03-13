"""
# Create a virtual environment
python -m venv erisk_2025
source erisk_2025/bin/activate  # On Windows: erisk_2025\Scripts\activate

# Install required packages
pip install langchain langchain-openai langchain-core langchain_community docx2txt pypdf  langchain_chroma sentence_transformers streamlit
"""
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional

def load_and_process_data(file_path: str, 
                          chunk_size: int = 1000, 
                          chunk_overlap: int = 200) -> List[Document]:
    """
    Load data from MentalChat16K CSV file, convert to documents and split into chunks.
    
    Args:
        file_path: Path to the CSV file
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks ready for embedding
    """
    # Load raw documents
    raw_documents = load_csv_documents(file_path)
    
    # Define text splitting configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # Split documents into chunks
    splits = text_splitter.split_documents(raw_documents)
    
    return splits

def load_csv_documents(file_path: str) -> List[Document]:
    """
    Load documents from CSV file with input/output pairs.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of Document objects
    """
    documents = []
    df = pd.read_csv(file_path)  # Load CSV into DataFrame
    
    # Ensure required columns exist
    if 'input' in df.columns and 'output' in df.columns:
        for _, row in df.iterrows():
            content = f"Input: {row['input']}\nOutput: {row['output']}"
            
            # Create metadata with depression indicators if available
            metadata = {}
            if 'depression_indicators' in df.columns:
                metadata['depression_indicators'] = row['depression_indicators']
            if 'severity' in df.columns:
                metadata['severity'] = row['severity']
                
            documents.append(Document(page_content=content, metadata=metadata))
    else:
        print("Error: CSV file must contain 'input' and 'output' columns.")
    
    return documents

def load_document_from_file(file_path: str) -> Optional[List[Document]]:
    """
    Load document from a file (PDF, DOCX, TXT)
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of Document objects or None if loading fails
    """
    import os
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    
    try:
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Select appropriate loader
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        elif ext in ['.txt', '.md']:
            loader = TextLoader(file_path)
        else:
            print(f"Unsupported file format: {ext}")
            return None
            
        # Load document
        return loader.load()
    except Exception as e:
        print(f"Error loading document {file_path}: {str(e)}")
        return None

# import pandas as pd
# from langchain_community.document_loaders import DataFrameLoader

# # # Load the synthetic dataset
# # df = pd.read_csv('data/MentalChat16K/Synthetic_Data_10K.csv')

# import pandas as pd
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from typing import List

# # Define text splitting configuration
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len
# )

# # Load CSV file and extract relevant columns
# def load_csv_documents(file_path: str) -> List[Document]:
#     documents = []
#     df = pd.read_csv(file_path)  # Load CSV into DataFrame
    
#     # Ensure required columns exist
#     if 'input' in df.columns and 'output' in df.columns:
#         for _, row in df.iterrows():
#             content = f"Input: {row['input']}\nOutput: {row['output']}"
#             documents.append(Document(page_content=content))
#     else:
#         print("Error: CSV file must contain 'input' and 'output' columns.")
    
#     return documents

# # File path
# file_path = "data/MentalChat16K/Synthetic_Data_10K.csv"
# documents = load_csv_documents(file_path)

# print(f"Loaded {len(documents)} documents.")

# # Split documents into chunks
# splits = text_splitter.split_documents(documents)
# print(f"Split the documents into {len(splits)} chunks.")
