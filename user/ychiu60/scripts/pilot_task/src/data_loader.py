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

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, ArrayType
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional

def load_and_process_data_spark(file_path: str, 
                              chunk_size: int = 1000, 
                              chunk_overlap: int = 200,
                              spark: SparkSession = None) -> List[Document]:
    """
    Load data from MentalChat16K CSV file using PySpark, convert to documents and split into chunks.
    
    Args:
        file_path: Path to the CSV file
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        spark: SparkSession object
        
    Returns:
        List of document chunks ready for embedding
    """
    # Create SparkSession if not provided
    if not spark:
        from src.utils import create_spark_session
        spark = create_spark_session()
    
    # Load CSV data into a DataFrame
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Ensure required columns exist
    if "input" in df.columns and "output" in df.columns:
        # Create document text by combining input and output
        df = df.withColumn("content", 
                         spark.sql.functions.concat(
                             spark.sql.functions.lit("Input: "), 
                             col("input"), 
                             spark.sql.functions.lit("\nOutput: "), 
                             col("output")
                         ))
        
        # Define UDF to create Document objects
        @udf(returnType=StringType())
        def create_document_content(content, depression_indicators=None, severity=None):
            return content
        
        # Apply UDF
        df = df.withColumn("document_content", 
                         create_document_content(col("content"), 
                                              col("depression_indicators") if "depression_indicators" in df.columns else None,
                                              col("severity") if "severity" in df.columns else None))
        
        # Collect results
        rows = df.select("document_content", "depression_indicators", "severity").collect()
        
        # Convert to Document objects
        raw_documents = []
        for row in rows:
            metadata = {}
            if hasattr(row, "depression_indicators") and row.depression_indicators:
                metadata['depression_indicators'] = row.depression_indicators
            if hasattr(row, "severity") and row.severity:
                metadata['severity'] = row.severity
                
            raw_documents.append(Document(page_content=row.document_content, metadata=metadata))
    else:
        print("Error: CSV file must contain 'input' and 'output' columns.")
        raw_documents = []
    
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

