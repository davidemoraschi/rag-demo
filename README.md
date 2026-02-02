"# rag-demo" 

# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Then pull a small model
ollama pull llama3.2
ollama pull nomic-embed-text  # for embeddings

pip install langchain langchain-community langchain-chroma
pip install ollama chromadb
pip install pypdf gitpython
