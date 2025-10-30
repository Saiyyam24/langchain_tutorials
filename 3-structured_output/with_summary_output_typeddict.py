from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated
from pydantic import BaseModel
import os
load_dotenv()

google_api_key = os.getenv('Gemini_api_key')
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash',
                               api_key=google_api_key)
class Review(TypedDict):
    summary:str
    sentiment:str
    # key_themes:Annotated

structured_model = model.with_structured_output(Review)

Result = structured_model.invoke('''The performance is very good at the price range of ₹24k-₹26k. The phone comes with dual speaker so sound quality is very good. It has glass back with metal frame so the phone may feel uncomfortable at first but with time it is good in handling. The speed is good. Durability is better and the phone doesn't hang too much. The phone is verall value for money.
''')
# structured_model = model.with_structured_output(Review.__annotations__)

print(Result)

# google genai is not supported typeddict 