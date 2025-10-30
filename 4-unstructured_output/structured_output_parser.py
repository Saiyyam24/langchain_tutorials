from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
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

schema = [
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give 3 facts about the {topic} \n{format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic':'black-hole'})
print(result)


# prompt = template.invoke({'topic':'blackhole'})
# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)