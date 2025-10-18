import os
import sys

# --- Fix for Azure SQLite version issue ---
# Force Chroma to use the modern bundled SQLite (pysqlite3)
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception as e:
    print(f"SQLite override skipped or failed: {e}")

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


# --- Portfolio Class ---
class Portfolio:
    def __init__(self, file_path="app/resource/my_portfolio.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

        # Use Chroma with local persistence
        self.chroma_client = chromadb.Client(
            settings=chromadb.Settings(persist_directory="./chroma_db")
        )

        try:
            self.collection = self.chroma_client.get_or_create_collection(name="portfolio")
        except chromadb.errors.ValueError:
            self.collection = self.chroma_client.create_collection(name="portfolio")

    def load_portfolio(self):
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
                email = llm.write_mail(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    create_streamlit_app(chain, portfolio, clean_text)
