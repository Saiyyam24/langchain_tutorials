import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("Gemini_api_key")
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    temprature = 0.2,
    google_api_key = api_key
)

response = llm.invoke("capital of india")
print(response.content)