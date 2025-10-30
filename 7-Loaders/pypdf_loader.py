from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r'D:\study\Langchain\7-Loaders\2.1 Linear Regression.pdf')
doc = loader.load()
# print(doc)

print(doc[6].page_content)
print(doc[6].metadata)
