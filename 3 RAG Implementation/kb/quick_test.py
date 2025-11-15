from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load a document
loader = PyPDFLoader("documents/policies/sample.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

print(f"Loaded {len(documents)} pages")
print(f"Split into {len(chunks)} chunks")
print(f"First chunk preview: {chunks[0].page_content[:200]}...")
