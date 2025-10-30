from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
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
parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age,city of a fictional person\n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
# prompt = template.format()

# result = model.invoke(prompt)
# final_result =parser.parse(result.content)
chain = template | model | parser
result = chain.invoke({})
print(result)
print(type(result))