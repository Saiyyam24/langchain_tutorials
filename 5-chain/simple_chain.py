from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate five inresting intresting fact about {topic}',
    input_variables=['topic']
)
model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    api_key=os.getenv('Gemini_api_key')
)
parser = StrOutputParser()

chain = prompt|model|parser

result = chain.invoke({'topic':'cricket'})
print(result)


