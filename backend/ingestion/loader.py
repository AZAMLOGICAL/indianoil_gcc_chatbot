import os
import pdfplumber
import re
from typing import List, Dict

def extract_pages_with_text_and_tables(pdf_path : str) -> List[Dict]:
    """
    Extract text and table from each page of a pdf
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            tables = page.extract_tables()
            
            page_info = {
                "text" : text,
                "tables" : tables,
                "page_number" : page.number 
            }
            pages.append(page_info)
    return pages

def clean_pdf_pages_text(pages: List[Dict], header:str = "Indian Oil Corporation General Conditions of Contract\n"):
    """
    Clears headers/footers from each page text except in the beginning
    """
    cleaned_texts = []
    for page in pages[1:]:
        text = page.get('text')
        if not text:
            continue
        # Remove the recurring footer from the beginning
        text = re.sub(r'^' + re.escape(header), '', text)
        # Remove the page number from every page
        text = re.sub(r'\n\d+\s*$', '', text)
        # Replace the existing text with this 
        page['text'] = text
        # append the text 
        cleaned_texts.append(text)
    return cleaned_texts

def load_and_clean_as
        
        
