from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from typing import TypedDict,Literal
load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    api_key=os.getenv('Gemini_api_key')
)
class Feedback(BaseModel):
    sentiment : Literal['Postive','Negative'] = Field(description='Give the feedback review Positive and Negative')
parser2 = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(
    template='classify the sentence rather sentence is postive or negative from feedback text \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

parser = StrOutputParser()
classifier_chain = prompt1 | model1 | parser2
prompt2 = PromptTemplate(
    template = 'write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt2 = PromptTemplate(
    template = 'write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)




# branch_chain = RunnableBranch(
#     (condition1, chain1),
#     (condition2, chain2),
#     default chain
#     )




branch_chain = RunnableBranch(
    (lambda x:x['sentiment']=='Positive', prompt2|model1|parser),
    (lambda x:x['sentiment']=='Nositive', prompt2|model1|parser),
    RunnableLambda(lambda x:'Could not find sentiment')
    )

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback':'Smartphone is terrible, dont buy this'})
print(result)