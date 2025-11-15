from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CHROMA_PATH = "data/chroma_db"

def query_knowledge_base(question):
    """
    Query the knowledge base and return an answer with sources.
    """
    print(f"Question: {question}\n")
    
    # Initialize embeddings (must match what was used to create the DB)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    # Load the existing vector database
    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # Create a retriever
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
    )
    
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0
    )
    
    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Get the answer
    result = qa_chain.invoke({"query": question})
    
    print("Answer:")
    print(result["result"])
    print("\nSources:")
    
    for i, doc in enumerate(result["source_documents"], 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        print(f"{i}. {source} (Page {page})")
        print(f"   Preview: {doc.page_content[:150]}...\n")

if __name__ == "__main__":
    # Example queries
    questions = [
        "What is the company's remote work policy?",
        "How many vacation days do employees get?",
        "What are the IT security requirements for laptops?"
    ]
    
    for question in questions:
        query_knowledge_base(question)
        print("\n" + "="*80 + "\n")
