import os 
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
api_key = os.getenv("Gemini_api_key")
embedding = GoogleGenerativeAIEmbeddings(
    model = "models/text-embedding-004",
    google_api_key=api_key
)

vector = embedding.embed_query("what is langchain")
print(len(vector))
print(vector[:10])