from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "data/chroma_db"

def main():
    """
    Interactive query interface for the knowledge base.
    """
    # Initialize components
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    print("Company Knowledge Base - Interactive Mode")
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not question:
            continue
        
        result = qa_chain.invoke({"query": question})
        print(f"\nAnswer: {result['result']}\n")
        
        print("Sources:")
        for i, doc in enumerate(result["source_documents"], 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            print(f"  {i}. {source} (Page {page})")
        print("\n" + "-"*80 + "\n")

if __name__ == "__main__":
    main()
