from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence
import os 
load_dotenv()

google_api_key = os.getenv('Gemini_api_key')

prompt = PromptTemplate(template='write a joke about {topic}',
                        input_variables=['topic'])
prompt2 = PromptTemplate(template = 'Explain the following joke {text}',
                         input_variables=['text'])
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash',
                               api_key=google_api_key)

parser = StrOutputParser()
chain = RunnableSequence(prompt,model,parser,prompt2,model,parser)

print(chain.invoke({'topic':'Ai'}))