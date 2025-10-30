from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path=r'D:\soroniyan\sport.csv')
docs = loader.load()
print(docs[0])