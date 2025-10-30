from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
gemini_api_key = os.getenv("Gemini_api_key")
embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",
                                         google_api_key=gemini_api_key)

documents = [
    "Delhi is the capital of India",
    "Jaipur is the capital of rajasthan",
    "Banglore is the capital of canada",
    "Lucknow is the capital of Uttar-pradesh"
]
question = "what is the capital of canada"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(question)

score = cosine_similarity([query_embedding],doc_embedding)[0] # here we add [0] to change the dimension from 2d to 1d.

# print(list(score))
for index,sc in enumerate(score):
    print(index,sc)
# for idx, sc in enumerate(score):
#     print(f"{documents[idx]} -> similarity: {sc:.4f}")
