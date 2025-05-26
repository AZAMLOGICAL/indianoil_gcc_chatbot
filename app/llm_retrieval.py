# Import liraries
import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

import warnings
warnings.filterwarnings("ignore")

# Create a function to load existing vector store
def load_vector_store(embedding_id:str, file_name:str) -> FAISS:
    """
    Function to load the existing vector store using the specified file location and the 
    appropriate model id.
    
    Args:
        embedding_id (str) : the model that was used for creating the embeddings
        file_name(str) : the name of the file where the embeddings are stored at
        
    Returns:
        vectorstore(FAISS) : the vector store containing all the embeddings
    """
    # Load the embedding model
    embedding = OllamaEmbeddings(model=embedding_id)
    # Load the vector store
    file_path = os.path.join('vector_storage', file_name)
    vector_store = FAISS.load_local(
        file_path,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
    # Print the number of vectors in the vector stores
    print("The number of vectors in the vector stores are : ", vector_store.index.ntotal)
    # return the vector store
    return vector_store
# Function to return the llm chat model
def create_chat_llm(model_id:str, temperature:float=0.2) -> Ollama:
    """
    It takes the model id as input and returns the llm as the output
    Args:
        model_id(str) : the name of the model to be used as llm
        temperature(float) : the temperature to be maintained in the llm setting
    """
    llm_url = "http://localhost:11434"
    # Instantiate the llm model
    llm = Ollama(model=model_id, base_url=llm_url, temperature=temperature)
    # return the llm
    return llm

# Function to return the prompt template
def create_prompt_template():
    """
    It is used to return the langchain chat prompt template 
    
    Returns:
    prompt_template (lanhgchain chat prompt template) : a template containing the system prompt and the 
    
    """
    system_prompt = """
                    You are a helpful assistant who answers questions using the provided context. 
                    If the answer is not found within the contexzt, just say that the answer of the query is not found within the document.
                    """ 
    # prompt template from langchain
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt),
                                           ("human", "{context}\n\nQuestion: {input}")])
    # return the prompt template 
    return prompt_template

def create_multi_query_retriver(llm:Ollama, vector_store:FAISS, k:int=3) -> MultiQueryRetriever:
    """
    It returns a multi query retriver in which for a particular query
    
    Args:
        llm (Ollama) : Chat model used for using with the multi query retriever
        vector_store(FAISS) : vector store containing the knowledge house 
        k(int) : Number of two similar documents
        
    Returns:
        multi_query_retriever (MultiQueryRetriever) : retriever to query the documents and return the k most similar documents
    """
    # intantiate the retriver
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k" : 3}),
        llm=llm
    )
    return multi_query_retriever

def create_rag_chain(llm:Ollama, prompt_template, retriever:MultiQueryRetriever) -> create_retrieval_chain:
    """
    It returns the langchain rag chain
    Args:
        llm(Ollama) : The llm model to be used as chat model
        prompt_template(langchain chat prompt template) : the chat prompt template to be used as chat prompt template
        retriever(MultiQueryRetriever) : the retriever to extract the documents from the vector store
        
    Returns:
            chain(create_retrieval_chain) : the chain finally used to invoke 
    """
    # Create the question answer chain
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    # Create the retrievel chain
    chain = create_retrieval_chain(retriever, question_answer_chain)
    # return the chain
    return chain
# Write a function to chat with rag chatbot
def chat_with_rag_bot(chain, question:str) -> str:
    """
    Send the question to the retrieval based LLM chain and returns the model response
    
    Args:
        chain : the langchain retrieval chain object
        question(str) : the user's input question
        
    Returns :
        str : the model generated answer
    """
    # Create response by invoking the chain
    response = chain.invoke({"input" : question})
    return response["answer"]
    
    
    
    
    

