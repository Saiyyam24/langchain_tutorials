from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence,RunnableLambda,RunnableParallel,RunnablePassthrough,RunnableBranch
import os 
load_dotenv()

prompt1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template='summarize the following text \n {text}',
    input_variables=['text']
)

google_api_key = os.getenv('Gemini_api_key')
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash',
                               api_key=google_api_key)
parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1,model,parser)
branch_chain = RunnableBranch(
    (lambda x:len(x.split())>100, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough
)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

result = final_chain.invoke({'topic':'russia vs ukraine'})

print(result)