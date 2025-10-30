from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,TextLoader

pdf_loader = DirectoryLoader(
    path = r'D:\study\Langchain\7-Loaders\book_file',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

pdf_docs = pdf_loader.load()

txt_loader = DirectoryLoader(
    path=r"D:\study\Langchain\7-Loaders\book_file",
    glob="*.txt",
    loader_cls=lambda path: TextLoader(path,encoding='UTF-8')
)
txt_docs = txt_loader.load()
doc = txt_docs + pdf_docs
print(len(doc))
print(doc[32])

