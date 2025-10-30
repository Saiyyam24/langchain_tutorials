from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import os


load_dotenv()
gemini_api_key = os.getenv('Gemini_api_key')
model = ChatGoogleGenerativeAI(
    model = 'gemini-2.0-flash',
    api_key=gemini_api_key
)



# 1prompt -> detailed report

template1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables=['topic']
)

#  2prompt->summary
template2 = PromptTemplate(
    template = 'Write a summary in 5 lines of {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser 
result = chain.invoke({'topic':'black-hole'})

print(result)