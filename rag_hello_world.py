# rag_hello_world.py
from langchain_community.document_loaders import PyPDFLoader, GitLoader  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain_ollama import OllamaEmbeddings  # type: ignore
from langchain_ollama import Ollama  # type: ignore
from langchain_chroma import Chroma  # type: ignore
from langchain_core.runnables import RunnablePassthrough  # type: ignore
from langchain_core.prompts import PromptTemplate  # type: ignore
from langchain_core.output_parsers import StrOutputParser  # type: ignore
import os

# ============ 1. LOAD DOCUMENTS ============

documents = []

# Load PDFs
pdf_files = [
    "SQL Performance Explained.pdf",
    "The Definitive Guide to SQLite.pdf",
]  # <-- Your PDFs here
for pdf_path in pdf_files:
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
        print(f"✓ Loaded {pdf_path}")

# Load GitHub repo
repo_path = "./temp_repo"  # Local clone destination
repo_url = (
    "https://github.com/microsoft/Windows-classic-samples.git"  # <-- Your repo here
)

loader = GitLoader(
    clone_url=repo_url,
    repo_path=repo_path,
    branch="master",
    file_filter=lambda x: x.endswith((".py", ".md", ".txt", ".js")),
)
documents.extend(loader.load())
print(f"✓ Loaded GitHub repo")

# ============ 2. CHUNK DOCUMENTS ============

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"✓ Created {len(chunks)} chunks")

# ============ 3. CREATE VECTOR STORE ============

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # Persists to disk
)
print("✓ Vector store created")

# ============ 4. CREATE RAG CHAIN ============

llm = Ollama(model="llama3.2")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# ============ 5. QUERY! ============

while True:
    query = input("\n🔍 Ask a question (or 'quit'): ")
    if query.lower() == "quit":
        break

    response = qa_chain.invoke(query)
    print(f"\n📝 Answer: {response}")
