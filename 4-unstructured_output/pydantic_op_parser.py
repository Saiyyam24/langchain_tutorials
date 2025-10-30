from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
import os


load_dotenv()
# gemini_api_key = os.getenv('Gemini_api_key')
api_key = os.getenv("HUGGING_FACE_MODEL") 
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task = "task-generation",
    huggingfacehub_api_token=api_key
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description='name of a person')
    age: int = Field(description='age of the person')
    city: str = Field(description='Name of the city')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template= 'Generate the name, age and city of a fictional {place} person \n{format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
chain = template | model | parser
result = chain.invoke({'place':'India'})
print(result)

# prompt = template.invoke({'place':'India'})
# result = model.invoke(prompt)
# print(result.content)