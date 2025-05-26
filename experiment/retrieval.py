from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")

# remove the warnings
import warnings
warnings.filterwarnings('ignore')

# persist_directory = "../chroma_db_experiment"
# persist_directory = r"chroma_db_openai"
embedding_id = "mxbai-embed-large"

embedding = OllamaEmbeddings(model = embedding_id)

# Open AI embeddings
# embedding = OpenAIEmbeddings(
#     model = "text-embedding-3-small"
# )

print('The execution of scrpt started')

# vector_store = Chroma(
#     persist_directory = persist_directory,
#     embedding_function=embedding
# )

vector_store = FAISS.load_local("faiss_db_experiment/faiss.index",
                                embeddings=embedding,
                                allow_dangerous_deserialization=True)

# # To check wether the vector store is empty or not
# collection = vector_store._collection
num_vectors = vector_store.index.ntotal
print(f"Vector store contains {num_vectors} vectors")

system_prompt = """
You are a helpful assistant who answers questions using the provided context. 
If the answer is not found within the contexzt, just say that the answer of the query is not found within the document.
""" 

retriever = vector_store.as_retriever(search_kwargs = {'k' : 3})

prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                           ("human", "{context}\n\nQuestion: {input}")])

model_id = "llama3.2:1b"
llm_url = "http://localhost:11434"

llm = Ollama(model=model_id, base_url=llm_url, temperature=0.2)
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
#     # base_url="...",
#     # organization="...",
#     # other params...
# )

compressor = LLMChainExtractor.from_llm(llm)

compressor_retriever = ContextualCompressionRetriever(
    base_compressor = compressor, base_retriever = retriever
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(compressor_retriever, question_answer_chain)

question = input("Enter any question to ask from the chatbot?: ")

# results = retriever.get_relevant_documents("How the payment is done against the work?")

# print(results)

response_sample = chain.invoke({"input" : question})

print(f"Model response: ", response_sample["answer"])













