# load the document
from langchain_community.document_loaders import  PDFPlumberLoader
from langchain_core.documents import Document
import re
from tqdm import tqdm
import chromadb

import pdfplumber

import warnings
warnings.filterwarnings('ignore')


import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

pages = []

print("The loading of data starts")
with pdfplumber.open(r"document/GCC_e78f9d59-77bb-4de7-99f41730977875550_Shefali_Singh.pdf") as pdf:
    for page in pdf.pages:
        # Extract text
        text = page.extract_text()
        
        # Extract tables
        tables = page.extract_tables()
        
        page_info = {
            "text" : text,
            "tables" : tables,
            "page_number" : page.page_number
        }
        
        pages.append(page_info)

print("The loading of data ended")
        
for page in pages[1:]:
    # Extract the text of every page
    text = page['text']
    # remove the Indian Oil Corporation General Conditions of Contract\n from every text
    header = 'Indian Oil Corporation General Conditions of Contract\n'
    text = re.sub(r'^' + re.escape(header), '', text)
    # Remove footer page number if present
    text = re.sub(r'\n\d+\s*$', '', text)
    # Replace the page text with this one
    page['text'] = text
    
# Now combine the texts into one single list to check them afterwards
all_page_texts = []
for page in pages:
    if page['text']:
        all_page_texts.append(page['text'])

full_texts = "\n".join(all_page_texts)

# Use recursive text splitter to divide the text into chuncks
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

print("The chunking starts")
# Initialise the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Storage for all chunks
chunks = splitter.create_documents([full_texts])

# For chroma db 
# documents = [chunck.page_content for chunck in chunks]

# Create ids for chromadb
# Create IDs using the index
ids = [f"chunk_{i}" for i in range(len(chunks))]

print("The chunking ended")

# Create and import langchain ollamaembeddings anbd chroma db vector stores
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
embedding_id = "mxbai-embed-large"

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")

# OLLAMA embeddings
embed = OllamaEmbeddings(model=embedding_id)

## Google gemini embeddings
# embed = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07",
#                                      task_type="RETRIEVAL_DOCUMENT")

# Open AI embeddings
# embed = OpenAIEmbeddings(
#     model = "text-embedding-3-small"
# )


persist_directory = "faiss_db_experiment"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embed 
)

# Storing data in batches
# batch_size=100
# for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding documents"):
#     batch = chunks[i:i+batch_size]
#     if i == 0:
#         # Create the chromadb vector store
#         vectorstore = Chroma.from_documents(
#             documents=batch,
#             embedding=embed,
#             persist_directory='./chroma_db_openai'
#         )
#     else:
#         vectorstore.add_documents(batch)

vectorstore.save_local('faiss_db_experiment/faiss.index')

num_vectors = vectorstore.index.ntotal
print('The number of vectors in the collection count is :', num_vectors)


print('The emebedding data is saved into the faiss db database')









