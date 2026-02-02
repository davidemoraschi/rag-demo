# rag_hello_world.py
from langchain_community.document_loaders import PyPDFLoader, GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import os

# ============ 1. LOAD DOCUMENTS ============

documents = []

# Load PDFs
pdf_files = ["document1.pdf", "document2.pdf"]  # <-- Your PDFs here
for pdf_path in pdf_files:
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
        print(f"✓ Loaded {pdf_path}")

# Load GitHub repo
repo_path = "./temp_repo"  # Local clone destination
repo_url = "https://github.com/username/repo"  # <-- Your repo here

loader = GitLoader(
    clone_url=repo_url,
    repo_path=repo_path,
    branch="main",
    file_filter=lambda x: x.endswith((".py", ".md", ".txt", ".js"))
)
documents.extend(loader.load())
print(f"✓ Loaded GitHub repo")

# ============ 2. CHUNK DOCUMENTS ============

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"✓ Created {len(chunks)} chunks")

# ============ 3. CREATE VECTOR STORE ============

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Persists to disk
)
print("✓ Vector store created")

# ============ 4. CREATE RAG CHAIN ============

llm = Ollama(model="llama3.2")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# ============ 5. QUERY! ============

while True:
    query = input("\n🔍 Ask a question (or 'quit'): ")
    if query.lower() == 'quit':
        break
    
    response = qa_chain.invoke(query)
    print(f"\n📝 Answer: {response['result']}")