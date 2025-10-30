from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.schema.runnable import RunnableParallel
load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    api_key=os.getenv('Gemini_api_key')
)
api_key = os.getenv("HUGGING_FACE_MODEL") 
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task = "task-generation",
    huggingfacehub_api_token=api_key
)
model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short quiz question from following text \n {text}',
    input_variables=['text']  
)
prompt3 = PromptTemplate(
    template='Merge the provide notes and quiz in the document \n {notes} \n {quiz}',
    input_variables=['notes','quiz']
)
parser = StrOutputParser()

paraller_chain = RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = paraller_chain | merge_chain
text = '''What's the difference between LangChain and LangGraph?
AI Overview
LangChain is a broad framework for building LLM applications, offering tools for chains and agents. LangGraph, built by LangChain, is a specific tool within the ecosystem designed to create stateful, complex agent workflows by modeling them as graphs with loops and conditional logic, providing superior state management and control over dynamic execution compared to LangChain's more linear, sequential approach. 
LangChain 
Purpose: Provides a foundational framework and modular components to build various LLM-powered applications, like chatbots, question-answering systems, and document summarizers. 
Structure: Centers around the concept of "chains," which are sequential pipelines where data flows in a one-way or Directed Acyclic Graph (DAG) structure. 
State Management: Handles state linearly, passing memory forward from one step to the next, which is suitable for simpler processes but can struggle with long-term, persistent context. 
Use Cases: Ideal for building applications with simpler, sequential tasks, such as document retrieval, summarization, and basic question answering. 
LangGraph 
Purpose: A framework specifically for orchestrating complex, interactive, stateful agent and multi-agent workflows by using a graph-based architecture. 
Structure: Uses a graph structure that allows for cyclical paths, allowing nodes to loop back, revisit previous states, and manage conditional branching. 
State Management: Treats state as a first-class citizen, with shared state accessible and modifiable by all nodes, enabling deep collaboration between agents and robust context retention. 
Use Cases: Better suited for building dynamic systems that require ongoing interaction, such as complex task automation, research assistants, or multi-agent systems that need to adapt and maintain context over long periods. 
In summary, if you need to build a simple, sequential LLM application, LangChain is sufficient. However, for complex, multi-agent systems that require sophisticated, stateful interactions and dynamic control flow, LangGraph provides the specialized architecture to handle these complexities. '''
result = chain.invoke(text)
print(result)