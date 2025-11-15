from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Verify setup
print("LangChain setup successful!")
print(f"API Key loaded: {bool(os.getenv('OPENAI_API_KEY'))}")
