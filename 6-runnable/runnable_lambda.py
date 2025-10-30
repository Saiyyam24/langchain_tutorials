from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence,RunnableLambda,RunnableParallel,RunnablePassthrough
import os 
load_dotenv()

def word_count(sentence):
    n = sentence.split()
    return len(n)


google_api_key = os.getenv('Gemini_api_key')

prompt = PromptTemplate(template='write a joke about {topic}',
                        input_variables=['topic'])
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash',
                               api_key=google_api_key)

parser = StrOutputParser()

joke_generate = RunnableSequence(prompt,model,parser)

result = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_generate,result)
print(final_chain.invoke({'topic':'Ai'}))