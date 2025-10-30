from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    api_key = os.getenv('Gemini_api_key')
)
messages = [
    SystemMessage(content='You are a helpful asistant'),

    HumanMessage(content='Tell me about Langchain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)