from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    api_key = os.getenv('Gemini_api_key')
)
chat_history = [
    SystemMessage(content='You atr a helpful ai assistant')
]
while True:
    user_input = input('You:')
    chat_history.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:",result.content)

print(chat_history)
