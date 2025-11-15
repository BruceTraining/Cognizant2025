from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
DOCUMENTS_PATH = "documents/policies"
CHROMA_PATH = "data/chroma_db"

def build_knowledge_base():
    """Load documents, chunk them, and store in Chroma."""
    print("Step 1: Loading documents...")
    
    # Load all PDF files from the documents directory
    loader = DirectoryLoader(
        DOCUMENTS_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages")
    
    print("\nStep 2: Splitting documents into chunks...")
    
    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")
    
    print("\nStep 3: Creating embeddings and storing in Chroma...")
    
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    # Create Chroma vector database
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    vector_count = vectordb._collection.count()
    print(f"\nSuccess! Created database with {vector_count} vectors")
    print(f"Database location: {CHROMA_PATH}")
    
    return vectordb

if __name__ == "__main__":
    build_knowledge_base()
