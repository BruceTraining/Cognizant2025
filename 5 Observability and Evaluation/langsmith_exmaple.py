from langsmith import traceable
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables (including LangSmith config)
load_dotenv()

@traceable(name="retrieve_documents", tags=["retrieval"])
def retrieve_relevant_chunks(query: str, vectordb: Chroma, k: int = 5):
    """
    Retrieve relevant document chunks from the vector database.
    This function will appear as a custom span in LangSmith traces.
    """
    results = vectordb.similarity_search(query, k=k)
    return results

@traceable(name="validate_response", tags=["validation", "quality-check"])
def validate_agent_response(response: str, required_keywords: list = None):
    """
    Validate that the agent response meets quality criteria.
    Failures here will be tracked separately in LangSmith.
    """
    if not response or len(response.strip()) < 10:
        raise ValueError("Response too short or empty")
    
    if required_keywords:
        missing = [
            kw for kw in required_keywords 
            if kw.lower() not in response.lower()
        ]
        if missing:
            return {"valid": False, "missing_keywords": missing}
    
    return {"valid": True, "response_length": len(response)}

@traceable(
    name="process_user_query",
    tags=["main-workflow"],
    metadata={"version": "1.0", "environment": "production"}
)
def process_user_query(query: str, vectordb: Chroma):
    """
    Main workflow function that orchestrates the RAG pipeline.
    All nested function calls will appear as child spans.
    """
    # Retrieve relevant documents (creates nested span)
    chunks = retrieve_relevant_chunks(query, vectordb, k=3)
    
    # Build context from chunks
    context = "\n\n".join([doc.page_content for doc in chunks])
    
    # Generate response (you would call your LLM here)
    response = (
        f"Based on {len(chunks)} documents: [Answer would go here]"
    )
    
    # Validate response (creates nested span)
    validation = validate_agent_response(response)
    
    return {
        "query": query,
        "response": response,
        "chunks_used": len(chunks),
        "validation": validation
    }

# Usage example
if __name__ == "__main__":
    CHROMA_PATH = "data/chroma_db"
    
    # Load existing vector database
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # Process a query - creates a complete trace in LangSmith
    result = process_user_query(
        "What are the key policies regarding data privacy?",
        vectordb
    )
