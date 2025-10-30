from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional
from pydantic import BaseModel,Field
import os
load_dotenv()

google_api_key = os.getenv('Gemini_api_key')
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash',
                               api_key=google_api_key)
class Review(BaseModel):
    key_theme:list[str] = Field(description="discuss all key theme discussed")
    summary:str = Field(description="Breif summary of review")
    sentiment:str = Field(description="Positive,negative,neutral")
    pros:Optional[list[str]] = Field(default=None,description="Write all the pros")
    con:Optional[list[str]] = Field(default=None,description="write all the cons")

structured_model = model.with_structured_output(Review)

Result = structured_model.invoke('''The performance is very good at the price range of ₹24k-₹26k. The phone comes with dual speaker so sound quality is very good. It has glass back with metal frame so the phone may feel uncomfortable at first but with time it is good in handling. The speed is good. Durability is better and the phone doesn't hang too much. The phone is verall value for money.
''')
# structured_model = model.with_structured_output(Review.__annotations__)

print(Result.summary)
