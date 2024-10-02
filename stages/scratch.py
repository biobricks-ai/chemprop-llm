from joblib import Memory
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

import time
import itertools
import io
import numpy as np, pandas as pd
import json
import os
import biobricks as bb
import pandas as pd
import pathlib
import shutil
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys
sys.path.append('./')

import stages.utils.bing_search as bing_search
import stages.utils.scraperapi as scraperapi
import stages.utils.chatgpt as chatgpt

crawler = scraperapi.WebCrawler(num_workers=10, scraperapi_key=os.getenv('SCRAPERAPI_KEY'), max_queue_size=10)

# Extract relevant text snippets around compound name or CAS number mentions
def text_blobs(text, compound_name, casrn):
    text_lower = text.lower()
    patterns = [re.escape(compound_name.lower()), re.escape(casrn.lower())]
    indices = [m.start() for p in patterns for m in re.finditer(p, text_lower)]
    return [text[max(0, i-500):min(len(text), i+500)] for i in sorted(indices)]
    
def pdf_to_text(content):
    return '\n'.join([page.extract_text() for page in PdfReader(io.BytesIO(content)).pages])

def html_to_text(content):
    return BeautifulSoup(content, 'html.parser').get_text()

# score a text blob based on how relevant it is to the compound name and casrn
def blobscore(blob, compound_name, casrn):
    text_lower = blob.lower()
    score = sum([
        compound_name.lower() in text_lower or casrn.lower() in text_lower,
        sum(term in text_lower for term in ['acute', 'inhalation', 'toxicity', 'lc50']),
        2 if 'ppm' in text_lower else 1 if 'mg/l' in text_lower or 'mg/m3' in text_lower else 0,
        '4 hour' in text_lower or '4-hour' in text_lower,
        2 * sum(phrase in text_lower for phrase in ['inhalation lc50', 'acute inhalation toxicity']),
        -1 if len(blob) < 100 else 0
    ])
    return score

# get good text blobs from a url based on having relevant terms to the args
def getblobs(url, compound_name, casrn):
    task = scraperapi.ScrapeTask(url, autoparse=True, binary=url.endswith('.pdf'), ultra_premium=True)
    res = crawler._scrape(task)
    text = pdf_to_text(res.content) if url.endswith('.pdf') else html_to_text(res.text)
    blobs = text_blobs(text, compound_name, casrn)
    return blobs

def build_prompt(compound_name, casrn):
    query = f"\"{compound_name}\" \"acute\" \"inhalation\" \"LC50\" \"ppm\""
    urls = [url['url'] for url in bing_search.bing_search(query, count=10)]
    
    blobs = list(itertools.chain.from_iterable(getblobs(url, compound_name, casrn) for url in urls))
    scores = [blobscore(blob, compound_name, casrn) for blob in blobs]
    
    topblobs = sorted(zip(blobs, scores), key=lambda x: x[1], reverse=True)[:5]
    blobtext = '\n\n'.join([blob for blob, score in topblobs])

    prompt = f"""use what you know, and the below text blobs to estimate 
    the 4-hour human inhalation LC50 value of {compound_name} with cas {casrn} in ppm.\n{blobtext}"""
    
    return prompt
