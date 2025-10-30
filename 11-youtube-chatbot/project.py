import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()
gemini_api_key = os.getenv('Gemini_api_key')
ytt_api = YouTubeTranscriptApi()

embedding = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004',
                                         google_api_key=gemini_api_key)

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash',
                               api_key=gemini_api_key)
video_id = 'o50N3-OaGdM'
result = ytt_api.fetch(video_id,languages=['en'])
transcript = "".join(chunk.text for chunk in result)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                          chunk_overlap=200
                                        )
chunk = splitter.create_documents(transcript)
vector_store = FAISS.from_documents(chunk,embedding)
retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs={"k":4})

res =retriever.invoke("what is leopard")
print