import os 
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from dotenv import load_dotenv
load_dotenv()

loader = PyPDFLoader(r'D:\study\Langchain\9-vector_stores\country-information-report-india.pdf')
docs = loader.load()

embedding = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004',
                                         google_api_key=os.getenv('Gemini_api_key'))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

split_docs = text_splitter.split_documents(docs)

connection = "postgresql+psycopg://postgres:Nupur%40123@localhost:5432/demo_db"

vector_store = PGVector(
    embeddings=embedding,
    collection_name='india_docs',
    connection=connection,
    use_jsonb=True
)
vector_store.add_documents(split_docs)