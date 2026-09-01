from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import groq
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="EJT AI Core")

print("Memuat model BAAI/bge-small-en-v1.5...")
# Model ini akan diunduh otomatis jika belum ada di cache lokal
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.get("/")
def read_root():
    return {"status": "FastAPI berjalan, model BAAI siap digunakan!"}