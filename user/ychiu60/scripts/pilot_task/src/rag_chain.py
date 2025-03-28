from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from typing import List, Dict, Any

from langchain_community.llms import HuggingFaceEndpoint, HuggingFaceHub, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline




def create_rag_chain(vectorstore: Chroma, 
                   model_name: str = "google/flan-t5-base",  # A smaller model that works locally
                   temperature: float = 0) -> Dict[str, Any]:
    """
    Create a RAG chain for depression detection using local models.
    
    Args:
        vectorstore: Vector store for document retrieval
        model_name: HuggingFace model name
        temperature: LLM temperature
        
    Returns:
        Dictionary containing chain components
    """
    # Initialize local language model
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 512, "temperature": temperature}
    )
    
    output_parser = StrOutputParser()
    
    # Create query transformation prompt
    query_transform_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", """Given the conversation history and the latest user input, formulate a search query that would help retrieve relevant information to detect signs of depression. 

Focus on:
1. Emotional cues (sadness, hopelessness, emptiness)
2. Negative thought patterns (worthlessness, guilt, pessimism)
3. Behavioral indicators (sleep issues, appetite changes, loss of interest)
4. Energy levels and fatigue
5. Suicidal ideation or thoughts of death

Create a search query that will find relevant examples from the database that match these potential patterns in the user's message.""")
    ])
    
    # Create query transformation chain
    query_transform_chain = query_transform_prompt | llm | output_parser
    
    # Set up the retriever with contextual compression
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    # Create conversation prompt
    conversation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a mental health assistant trained to detect signs of depression through conversation. Your role is to engage in a supportive, empathetic conversation while quietly assessing for depression indicators.

DEPRESSION INDICATORS TO MONITOR FOR:
1. Persistent sad, anxious, or "empty" mood
2. Loss of interest or pleasure in hobbies and activities
3. Changes in sleep patterns (insomnia or oversleeping)
4. Changes in appetite or weight
5. Decreased energy, fatigue, feeling "slowed down"
6. Feelings of worthlessness, guilt, helplessness
7. Difficulty thinking, concentrating, or making decisions
8. Thoughts of death or suicide, or suicide attempts
9. Restlessness, irritability
10. Physical symptoms (headaches, digestive issues) with no clear cause

CONVERSATION GUIDELINES:
- Be warm, empathetic, and conversational
- DO NOT directly state that the user may have depression
- DO NOT attempt to diagnose
- Respond naturally to what the user says
- Avoid clinical language in your visible response
- If the user mentions self-harm or suicide, gently encourage them to seek professional help

After your normal conversational response, include a HIDDEN assessment section formatted exactly as follows:

Depression indicators: [list any detected signs, comma separated]
Severity estimate: [none/low/moderate/high]
Recommended action: [continue monitoring/professional referral needed]

The hidden assessment section should only be visible to healthcare professionals monitoring this chat.

Refer to the context from retrieved documents to inform your assessment, but maintain a natural conversation.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Context information from similar cases: {context}"),
    ])
    
    # Return components dictionary
    return {
        "llm": llm,
        "output_parser": output_parser,
        "query_transform_chain": query_transform_chain,
        "retriever": compression_retriever,
        "conversation_prompt": conversation_prompt
    }

def run_rag_chain(chain_components: Dict[str, Any], 
                 input_text: str, 
                 chat_history: List) -> str:
    """
    Run the RAG chain on a user input with chat history.
    
    Args:
        chain_components: Dictionary of chain components
        input_text: User input text
        chat_history: Chat history
        
    Returns:
        Response text
    """
    # Extract components
    llm = chain_components["llm"]
    output_parser = chain_components["output_parser"]
    query_transform_chain = chain_components["query_transform_chain"]
    retriever = chain_components["retriever"]
    conversation_prompt = chain_components["conversation_prompt"]
    
    # Transform query based on conversation history
    transformed_query = query_transform_chain.invoke({
        "input": input_text,
        "chat_history": chat_history
    })
    
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(transformed_query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Generate conversation messages
    messages = conversation_prompt.invoke({
        "input": input_text,
        "chat_history": chat_history,
        "context": context
    })
    
    # Run through LLM
    response = llm.invoke(messages)
    
    # Parse output
    return output_parser.invoke(response)


