from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
prompt = PromptTemplate(
    template = "Answer the folloing {question} from the following text: \n {text}",
    input_variables=['question','text']
)

model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    api_key=os.getenv('Gemini_api_key')
)
parser = StrOutputParser()

url = "https://www.flipkart.com/microsoft-xbox-series-s-512-gb/p/itm13c51f9047da8"
loader = WebBaseLoader(url)

docs= loader.load()
print(docs)
chain = prompt | model |parser

result = chain.invoke({'question':'Please give me specification details','text':docs[0].page_content})
print(result)