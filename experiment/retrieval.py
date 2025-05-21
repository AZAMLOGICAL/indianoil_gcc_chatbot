from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# remove the warnings
import warnings
warnings.filterwarnings('ignore')

persist_directory = "../chroma_db_experiment"
embedding_id = "mxbai-embed-large"

embedding = OllamaEmbeddings(model = embedding_id)

vector_store = Chroma(
    persist_directory = persist_directory,
    embedding_function=embedding
)

system_prompt = """
You are the lawyer and author of the contract agreement written in the organisation Indian Oil Corporation Limited.
Your job is to help the contractors and the officers of the organisation to do their job better by making them understand various clause agreements,
quote the necessary contract agreements if asked about the related contract agreements.
Only extract answers which are relevant to the question.
If no context is available, use the question as reference.
Context : {context}
"""

retriever = vector_store.as_retriever(search_kwargs = {'k' : 1})

prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                           ("human", "{input}")])

model_id_gemma = "gemma2:2b"
llm_url = "http://localhost:11434"

llm_gamma = Ollama(model=model_id_gemma, base_url=llm_url, temperature=0.2)

compressor = LLMChainExtractor.from_llm(llm_gamma)

compressor_retriever = ContextualCompressionRetriever(
    base_compressor = compressor, base_retriever = retriever
)

question_answer_chain_gamma = create_stuff_documents_chain(llm_gamma, prompt)
chain_gamma = create_retrieval_chain(compressor_retriever, question_answer_chain_gamma)

question = input()

response_sample_gamma = chain_gamma.invoke({"input" : question})

print("Gamma response: ", response_sample_gamma["answer"])













