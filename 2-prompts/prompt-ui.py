from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
google_api_key = os.getenv('Gemini_api_key')
st.header('Research tool')

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=google_api_key
)
paper_input = st.selectbox(
    "Select Research Paper name",["Attention Is All You Need","BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding","Language Models are Few-Shot Learners"]
)
template = PromptTemplate(
    template='''
        you are a good reader, who summerizes the given research paper given title is  {paper_input} for beginer level
        person, In one to two paragraph.
        1. Mathematical details:
           - All claculations and mathematical part should be clear and easy to understand.
        2. Analogies:
           - USe relatable Analogies to simplify complex ideas
        If certain information is not available in paper just reply "Insufficient information available in the paper"
        instead of guessing.
    ''',
    input_variables='paper_input'
)

prompt = template.invoke({'paper_input':paper_input})
if st.button("submit"):
    if prompt:
        answer = model.invoke(prompt)
        st.write(answer.content)
    else:
        st.write("Please select research paper")