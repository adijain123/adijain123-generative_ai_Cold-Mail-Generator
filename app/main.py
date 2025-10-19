import os
import sys
import shutil

# --- Fix for Azure SQLite version issue ---
# Force Chroma to use the modern bundled SQLite (pysqlite3)
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    print("Successfully overridden sqlite3 with pysqlite3")
except ImportError:
    print("pysqlite3 not available, using system sqlite3")
except Exception as e:
    print(f"SQLite override failed: {e}")

# --- Regular imports ---
import streamlit as st
import pandas as pd
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
import chromadb

from chains import Chain
from utils import clean_text

# --- Environment setup ---
load_dotenv()

if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "Cold-Mail-Generator/1.0"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"


# --- Cache Management ---
def clear_chroma_cache():
    """Clear ChromaDB cache to fix corrupted ONNX models"""
    cache_paths = [
        os.path.expanduser("~/.cache/chroma"),
        "/root/.cache/chroma",
        "./chroma_cache"
    ]
    
    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            try:
                shutil.rmtree(cache_path)
                print(f"Cleared ChromaDB cache at {cache_path}")
            except Exception as e:
                print(f"Failed to clear cache at {cache_path}: {e}")


# --- Portfolio Class ---
class Portfolio:
    def __init__(self, file_path="app/resource/my_portfolio.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

        # Initialize ChromaDB with error handling for ONNX model issues
        self.chroma_client = None
        self.collection = None
        self._initialize_chroma()

    def _initialize_chroma(self):
        """Initialize ChromaDB with proper error handling for ONNX model issues"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Use Client() instead of PersistentClient() for better Azure compatibility
                try:
                    self.chroma_client = chromadb.Client()
                except Exception:
                    # Fallback to PersistentClient if Client() fails
                    self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

                # Use get_or_create_collection to avoid race conditions
                self.collection = self.chroma_client.get_or_create_collection(name="portfolio")
                print("ChromaDB initialized successfully")
                return
                
            except Exception as e:
                error_msg = str(e).lower()
                if "onnx" in error_msg or "protobuf" in error_msg or "model" in error_msg:
                    print(f"ONNX model error detected (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        print("Clearing ChromaDB cache and retrying...")
                        clear_chroma_cache()
                        # Wait a moment before retry
                        import time
                        time.sleep(2)
                    else:
                        print("Failed to initialize ChromaDB after clearing cache")
                        raise e
                else:
                    print(f"ChromaDB initialization error: {e}")
                    raise e

    def load_portfolio(self):
        if not self.collection:
            print("ChromaDB collection not initialized")
            return
            
        if self.collection.count() == 0:
            for _, row in self.data.iterrows():
                self.collection.add(
                    documents=[row["Techstack"]],
                    metadatas=[{"links": row["Links"]}],
                    ids=[str(uuid.uuid4())]
                )

            if hasattr(self.chroma_client, "persist"):
                self.chroma_client.persist()

    def query_links(self, skills):
        if not self.collection:
            print("ChromaDB collection not initialized")
            return []
            
        result = self.collection.query(query_texts=skills, n_results=2)
        return result.get('metadatas', [])


# --- Streamlit App ---
def create_streamlit_app(llm, portfolio, clean_text):
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    st.title("📧 Cold Mail Generator")

    url_input = st.text_input(
        "Enter a URL:",
        value="https://jobs.careers.microsoft.com/global/en/search?l=en_us&pg=1&pgSz=20&o=Relevance&flt=true&ref=cms"
    )
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                # If no links found due to ChromaDB issues, use empty list
                if not links:
                    links = []
                email = llm.write_mail(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")
            # Show additional help for ONNX model errors
            if "onnx" in str(e).lower() or "protobuf" in str(e).lower():
                st.info("💡 **Tip**: This appears to be a model loading issue. The application will attempt to clear the cache and retry automatically.")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    create_streamlit_app(chain, portfolio, clean_text)
