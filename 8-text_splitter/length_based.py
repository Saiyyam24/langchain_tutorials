from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader(r'D:\study\Langchain\8-text_splitter\APZNZA~1.PDF')
docs = loader.load()

text =  ''' 
  Model exploration and hosting 
Google Cloud provides a set of state-of-the-art foundation models through Vertex AI, including Gemini. You can also deploy a third-party model to either Vertex AI Model Garden or self-host on GKE or Compute Engine.
Prompt design and engineering 
Prompt design is the process of authoring prompt and response pairs to give language models additional context and instructions. After you author prompts, you feed them to the model as a prompt dataset for pretraining. When a model serves predictions, it responds with your instructions built in.
Grounding and RAG 
Grounding connects AI models to data sources to improve the accuracy of responses and reduce hallucinations. RAG, a common grounding technique, searches for relevant information and adds it to the model's prompt, ensuring output is based on facts and up-to-date information.
  '''
splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=','
)
result = splitter.split_documents(docs)
print(result[10].page_content)
