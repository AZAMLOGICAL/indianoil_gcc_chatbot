# Import libraries
import re
import logging
import time
import pdfplumber
from typing import List, Dict
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaEmbeddings

import warnings
warnings.filterwarnings('ignore')

logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Get the logger
logger = logging.getLogger()

# Create function to the load the text from the pdf and return the list of dictionary containing page info
def load_pdf_data(file_name:str) -> List[Dict]:
    """
    It is a function to load the contents of each and every page of the pdf file
    
    Args:
        file_name (str) : The file name of the pdf file to be imported
        
    Returns:
        pages(List[Dict]) : A list with each element containing the information about each page
    """
    # place holder to save the information of each page
    pages = []
    
    # Open the file
    file_path = os.path.join('document', file_name)
    with pdfplumber.open(file_path) as pdf:
        # Loop through each page in pdf
        for page in pdf.pages:
            # Extract text
            text = page.extract_text()
            # Extract tables
            tables = page.extract_tables()
            # Create dictionary for each page
            page_info = {
                "text" : text,
                "tables" : tables,
                "page_number" : page.page_number
            }
            # Append the page info to the list
            pages.append(page_info)
    # return the list of dictionaries containing the page information
    return pages

# Create a function to remove the header and footer of each and every text from each file
def pre_process_pdf_text_data(pages : List[Dict]) -> List[Dict]:
    """
    It is a function to clean the header and footers from each and every page of the extracted text from the pdf
    
    Args:
        pages(List[Dict]) : uncleaned text of the pdf extracted
        
    Returns:
        pages(List[Dict]) : cleaned text of the pdf extracted
    """
    # Loop through each page except the first one
    for page in pages[1:]:
        # Extract the text of every page
        text = page['text']
        # remove the Indian Oil Corporation General Conditions of Contract\n from every text
        header = 'Indian Oil Corporation General Conditions of Contract\n'
        # Remove the header
        text = re.sub(r'^' + re.escape(header), "", text)
        # Remove the page number from every page text
        text = re.sub('\n\d+\s*$', '', text)
        # Replace the page text with the new cleaned one
        page['text'] = text
    # Return the list of dictionaries
    return pages

# Create a function to  create a full text based on the text from each pages
def create_one_text(pages: List[Dict]) -> str:
    """
    This function takes the list containing dictionary element containing information of every page 
    and returns a full text info containing the text from each and evry page.
    
    Args:
        pages (List[Dict]) : a list containg the page info of every page in a dictionary
        
    Returns :
        text (str) : text containing all the texts of pdf pages combined
    """
    # Place holder to contain each page text
    all_page_texts = []
    # loop through each element of the list
    for page in pages:
        # Check to see wether that page has text
        if page['text']:
            # Append the text to each page
            all_page_texts.append(page['text'])
    # Return the text
    return " ".join(all_page_texts)

# Create a function to convert text to different chunks
def create_chunks_text(text:str, chunk_size:int = 1000, chunk_overlap:int=200) -> List[Document]:
    """
    It is a function to convert a big text into a list of chunks each containing a document
    
    Args:
        text(str) : text containing all the texts from each page of the pdf
        chunk_size(int) : the size of each chunk
        chunk_overlap(int) : the overlap to be given for creating chunks
        
    Returns:
        List[Document] : list containing the document corresponding to each and every chunk
    """
    # Inititialise the splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # Create the list of documents
    chunks = splitter.create_documents([text])
    # return the list of documents(chunks)
    return chunks

# Function to create the embeddings
def create_vector_store(chunks:List[Document], model_id:str, embedding_file_name:str) -> None:
    """
    It is a function to convert the chunks into embeddings and storing it into a vector store
    
    Args:
        model_id(str) : embedding model to save the chunks into vector embeddings
        embedding_file_name(str) : the name of the file to store the embedding of the file
        
    Returns:
        None : it does not return anything but saves the embedding into the specified file name in a vector database
    """
    # Intialise the ollama embedding model
    embed = OllamaEmbeddings(model=model_id)
    # Create the vector store
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embed
    )
    # Save the vcetor store 
    vector_storage_folder = "vector_storage"
    vector_store_path = os.path.join(vector_storage_folder, embedding_file_name)
    # Save it to local storage
    vectorstore.save_local(vector_store_path)
    # print the number of vectors in the vector store
    num_vectors = vectorstore.index.ntotal
    print('The number of vectors in the vector stores are :', num_vectors)
    
    
    
    
        
        
    
    


