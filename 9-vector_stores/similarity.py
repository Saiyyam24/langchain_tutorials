from langchain_postgres import PGVector
from dotenv import load_dotenv
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004',
                                         google_api_key=os.getenv('Gemini_api_key'))

connection = "postgresql+psycopg://postgres:Nupur%40123@localhost:5432/demo_db"

vector_store = PGVector(
    embeddings=embedding,
    collection_name='india_docs',
    connection=connection,
    use_jsonb=True
)



query = "How many diffrent religion are there in India"
results = vector_store.similarity_search(query, k=3)

model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    api_key=os.getenv('Gemini_api_key')
)

prompt = PromptTemplate(
    template = ''' 
        You are a query assisstant, who answer the question from particular given data 
        data : {context}
        question : {question}
    ''',
    input_variables=['context','question']
)


parser = StrOutputParser()

chain = prompt | model | parser

chat_bot_result = chain.invoke({'context':results,'question':query})
print(chat_bot_result)