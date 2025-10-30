from langchain_community.document_loaders import TextLoader

loader = TextLoader(r'D:\study\Langchain\7-Loaders\sept_1_2ex.txt','UTF-8')
doc = loader.load()
print(doc[0].page_content)
print(type(doc))