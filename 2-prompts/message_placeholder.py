from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# chat Template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []

with open(r'D:\study\Langchain\2-prompts\chat_history.txt') as f:
    chat_history.extend(f.readlines()) 

# create prompt
prompt = chat_template.invoke({
    'chat_history':chat_history,
    'query':'where is my refund'
})

print(prompt)
