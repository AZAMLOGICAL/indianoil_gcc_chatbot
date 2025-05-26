# Import the libraries
import streamlit as st
import sys

from controller import DefineControllerList
import vector_store as vs
import llm_retrieval 

import logging
# get the logger
logger = logging.getLogger()

# set the level of logger
logger.setLevel(logging.INFO)
# Add stream handler to the logger
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


# Create a function to create the vector store
def create_vector_store(file_name:str, chunk_size:int, chunk_overlap:int, model_id:str, embedding_file_name:str):
    # Load the file
    logger.info("Loading of data started..")
    pages = vs.load_pdf_data(file_name)
    logger.info("Loading of pdf data completed")
    
    # Clean the pdf texts by removing unnecessary headers and footers
    logger.info("Preprocessing of text from pdf started...")
    pages = vs.pre_process_pdf_text_data(pages)
    logger.info("Preprocessing of text from pdf completed")
    
    # extract the text from all the pages
    logger.info("Combining of text from pdf started....")
    all_pages_text = vs.create_one_text(pages)
    logger.info("Combination of texts from each pdf completed")
    
    # Create chunks from the all page text
    logger.info("Chunking of data started .....")
    chunks = vs.create_chunks_text(all_pages_text, 1000, 200)
    logger.info("Chunking of data completed")
    
    # Create the vector store
    logger.info("Creation of vector store from text from pdfs started .....")
    vs.create_vector_store(chunks=chunks, model_id=model_id, embedding_file_name=embedding_file_name)
    logger.info("Vector store creation copmpleted **********************")
    
    
# Create a function for the creation of chain

def create_chain(model_id:str, embedding_file_name:str, chat_llm:str, temperature:float, k:int):
    logger.info("The retrieval chain started to create ......")
    # Load the vector store
    vector_store = llm_retrieval.load_vector_store(embedding_id=model_id, file_name=embedding_file_name)
    # Create the chat llm
    chat_llm = llm_retrieval.create_chat_llm(model_id=chat_llm, temperature=temperature)
    # Create the langchain chat prompt template
    chat_prompt_template = llm_retrieval.create_prompt_template()
    # Create the retriever
    multi_query_retriver = llm_retrieval.create_multi_query_retriver(llm=chat_llm, vector_store=vector_store, k=k)
    # create the rag chain
    chain = llm_retrieval.create_rag_chain(llm=chat_llm,prompt_template=chat_prompt_template,retriever=multi_query_retriver)
    logger.info("The retrieval chain completed ....")
    # return the chain
    return chain


if __name__ == '__main__':
    file_name = "GCC_e78f9d59-77bb-4de7-99f41730977875550_Shefali_Singh.pdf"
    model_id = "mxbai-embed-large"
    embedding_file_name = "faiss.index"
    chat_llm = "llama3.2:1b"
    temperature=0.2
    k=10
    # Read the controller list 
    wether_vector_store_to_create = DefineControllerList(sys.argv)
    # Code to create vector store 
    if wether_vector_store_to_create == "True":
        create_vector_store(file_name, chunk_size=1000, chunk_overlap=200, model_id=model_id, embedding_file_name=embedding_file_name)
    # Create the rag chain
    chain = create_chain(model_id=model_id,
                         embedding_file_name=embedding_file_name,
                         chat_llm=chat_llm,
                         temperature=temperature,
                         k=k)
    # Ask the question from the user
    question = input("Enter the question to be asked from the chatbot")
    # get the answer
    chatbot_answer = llm_retrieval.chat_with_rag_bot(chain, question)
    print(chatbot_answer)
    