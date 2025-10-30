import os 
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
api_key = os.getenv("Gemini_api_key")
embedding = GoogleGenerativeAIEmbeddings(
    model = "models/text-embedding-004",
    google_api_key=api_key
)
docs = [
    "Delhi is capital of india",
    "Kolkata is capital of west bengal",
]

vector = embedding.embed_documents(docs)
print(len(vector))
print(vector[:10])