# load the document
from langchain_community.document_loaders import  PDFPlumberLoader
import re

import pdfplumber

pages = []
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

# Initialise the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Storage for all chunks
chunks = splitter.create_documents([full_texts])

# Create and import langchain ollamaembeddings anbd chroma db vector stores
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
embedding_id = "mxbai-embed-large"

embed = OllamaEmbeddings(model=embedding_id)

# Create the chromadb vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embed,
    persist_directory='./chroma_db_experiment'
)

print('The emebedding data is saved into the chroma db database')









