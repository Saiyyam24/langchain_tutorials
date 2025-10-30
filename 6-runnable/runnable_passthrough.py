from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough
import os 
load_dotenv()


google_api_key = os.getenv('Gemini_api_key')
# passthrough = RunnablePassthrough()

prompt1 = PromptTemplate(template='write a joke about {topic}',
                        input_variables=['topic'])
prompt2 = PromptTemplate(template = 'Explain the following joke {text}',
                         input_variables=['text'])
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash',
                               api_key=google_api_key)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explain': RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)
print(final_chain.invoke({'topic':'cricket'}))
