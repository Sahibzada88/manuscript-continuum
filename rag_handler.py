from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_era_texts(century):
    """Load texts based on historical period"""
    era_map = {
        "14th": ["Canterbury_Tales.txt"],
        "18th": ["Robinson_Crusoe.txt", "Gullivers_Travels.txt"],
        "19th": ["Frankenstein.txt", "Pride_and_Prejudice.txt"]
    }
    
    texts = []
    for filename in era_map.get(century, ["default.txt"]):
        filepath = os.path.join("data", filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read()[:50000])  # Load first 50k characters
    return texts

def create_vector_store(century):
    """Create Chroma vector store for era"""
    texts = load_era_texts(century)
    if not texts:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    documents = text_splitter.create_documents(texts)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=f"manuscript_{century}",
        persist_directory="./chroma_db"
    )

def get_context(vector_store, query, k=3):
    """Retrieve relevant historical context"""
    if not vector_store:
        return "No historical data available for this era"
    return vector_store.similarity_search(query, k=k)