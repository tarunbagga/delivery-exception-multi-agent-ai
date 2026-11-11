# Notebook setup extracted and reformatted for publication. Python 3.12+.
# Execute in a shared notebook namespace before tools.py and preprocessing.py.
import os
import re
import csv
import json
import time
import sqlite3
import logging
import warnings
from collections import defaultdict
from typing import TypedDict, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langsmith import traceable
from pydantic import BaseModel, Field, model_validator
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 120)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('exception_pipeline')
print('Libraries imported successfully.')

with open('config.json', 'r') as file:
    config = json.load(file)
OPENAI_API_KEY = config.get('OPENAI_API_KEY')
OPENAI_API_BASE = config.get('OPENAI_API_BASE')
LANGCHAIN_TRACING_V2 = config.get('LANGCHAIN_TRACING_V2')
LANGCHAIN_API_KEY = config.get('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT = config.get('LANGCHAIN_PROJECT')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ['OPENAI_BASE_URL'] = OPENAI_API_BASE
if not LANGCHAIN_API_KEY or str(LANGCHAIN_API_KEY).lower().startswith('your'):
    LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2 = (None, 'false')
os.environ['LANGCHAIN_TRACING_V2'] = str(LANGCHAIN_TRACING_V2).lower()
if LANGCHAIN_API_KEY:
    os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
os.environ['LANGCHAIN_PROJECT'] = LANGCHAIN_PROJECT or 'last-mile-exception-agent'
print(f'OpenAI key loaded: {bool(OPENAI_API_KEY)} | Custom base URL: {bool(OPENAI_API_BASE)}')
print(f"LangSmith tracing: {os.environ['LANGCHAIN_TRACING_V2']} | Project: {os.environ['LANGCHAIN_PROJECT']}")

from langsmith import Client
try:
    langsmith_client = Client()
    LANGSMITH_ENABLED = os.environ.get('LANGCHAIN_TRACING_V2') == 'true' and bool(os.environ.get('LANGCHAIN_API_KEY'))
except Exception:
    langsmith_client, LANGSMITH_ENABLED = (None, False)
print('LangSmith tracing enabled' if LANGSMITH_ENABLED else 'LangSmith not configured - using local traces only')

GEN_MODEL = 'gpt-4o-mini'
EVAL_MODEL = 'gpt-4o'
gen_llm = ChatOpenAI(model=GEN_MODEL, temperature=0, max_retries=3, timeout=60)
eval_llm = ChatOpenAI(model=EVAL_MODEL, temperature=0, max_retries=3, timeout=60)
# These smoke tests make paid API calls when executed.
print('gen_llm  ->', gen_llm.invoke('Reply with the single word: ready').content)
print('eval_llm ->', eval_llm.invoke('Reply with the single word: ready').content)

import glob
import zipfile
REQUIRED_FILES = ['delivery_logs.csv', 'ground_truth.csv', 'customers.db', 'exception_resolution_playbook.pdf']

def locate_data_dir() -> str:
    """Find local dataset files. Extract only trusted dataset ZIP archives."""
    def has_all(d: str) -> bool:
        return all(os.path.exists(os.path.join(d, f)) for f in REQUIRED_FILES)
    for d in ['.', 'Data', 'data', 'Datasets_Last-Mile_Delivery']:
        if has_all(d):
            return d
    for zp in glob.glob('*.zip'):
        with zipfile.ZipFile(zp) as z:
            if any(name.endswith('delivery_logs.csv') for name in z.namelist()):
                z.extractall('Data')
                print(f'Extracted {zp} -> ./Data')
                break
    for d in ['Data'] + glob.glob('Data/*/') + glob.glob('*/'):
        if has_all(d):
            return d
    raise FileNotFoundError(f'Could not find {REQUIRED_FILES} in the working directory, a sub-folder, or a zip.')

DATA_DIR = locate_data_dir()
print(f'Data directory: {os.path.abspath(DATA_DIR)}')
DELIVERY_LOGS_PATH = os.path.join(DATA_DIR, 'delivery_logs.csv')
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, 'ground_truth.csv')
CUSTOMERS_DB_PATH = os.path.join(DATA_DIR, 'customers.db')
PLAYBOOK_PDF_PATH = os.path.join(DATA_DIR, 'exception_resolution_playbook.pdf')
for p in [DELIVERY_LOGS_PATH, GROUND_TRUTH_PATH, CUSTOMERS_DB_PATH, PLAYBOOK_PDF_PATH]:
    assert os.path.exists(p), f'Missing input file: {p}'
delivery_logs_df = pd.read_csv(DELIVERY_LOGS_PATH)
ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH, keep_default_na=False)
print(f"delivery_logs.csv : {delivery_logs_df.shape[0]} rows, {delivery_logs_df['shipment_id'].nunique()} unique shipments")
print(f'ground_truth.csv  : {ground_truth_df.shape[0]} rows')
display(delivery_logs_df)
profile = delivery_logs_df.groupby('shipment_id').agg(rows=('status_code', 'size'), duplicate_scans=('is_duplicate_scan', 'sum'), max_attempt=('attempt_number', 'max'), statuses=('status_code', lambda s: ', '.join(sorted(set(s)))))
display(profile)
print('Status code distribution:')
display(delivery_logs_df['status_code'].value_counts().to_frame('count'))
display(ground_truth_df[['shipment_id', 'is_exception', 'expected_resolution', 'expected_tone', 'should_escalate']])
with sqlite3.connect(CUSTOMERS_DB_PATH) as conn:
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].tolist()
    customers_df = pd.read_sql('SELECT * FROM customers', conn)
    lockers_df = pd.read_sql('SELECT * FROM lockers', conn)
print(f'Tables: {tables}')
print(f'customers: {len(customers_df)} rows | lockers: {len(lockers_df)} rows')
display(customers_df.assign(name=customers_df['name'].str[0] + '***'))
display(lockers_df)

import unicodedata
playbook_pages = PyPDFLoader(PLAYBOOK_PDF_PATH, extraction_mode='layout').load()
for doc in playbook_pages:
    text = unicodedata.normalize('NFKC', doc.page_content)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    doc.page_content = '\n'.join(line.strip() for line in text.splitlines()).strip()
SECTION_PATTERN = re.compile(r'^\s*(\d+)\.\s+([A-Z][A-Za-z /-]+)$', re.MULTILINE)
current_section = 'Introduction'
for doc in playbook_pages:
    doc.metadata['page'] = doc.metadata.get('page', 0) + 1
    match = SECTION_PATTERN.search(doc.page_content)
    if match:
        current_section = f'{match.group(1)}. {match.group(2).strip()}'
    doc.metadata['section'] = current_section
print(f'Loaded {len(playbook_pages)} pages from the playbook')
display(pd.DataFrame([{'page': d.metadata['page'], 'section': d.metadata['section'], 'chars': len(d.page_content), 'preview': d.page_content.strip()[:80].replace('\n', ' ')} for d in playbook_pages]))
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150, separators=['\n\n', '\n', '. ', ' '])
playbook_chunks = splitter.split_documents(playbook_pages)
for i, chunk in enumerate(playbook_chunks):
    chunk.metadata['chunk_id'] = i
    chunk.page_content = f"[{chunk.metadata['section']}]\n{chunk.page_content}"
print(f'{len(playbook_pages)} pages -> {len(playbook_chunks)} chunks')
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', encode_kwargs={'normalize_embeddings': True})
playbook_vectorstore = Chroma.from_documents(documents=playbook_chunks, embedding=embedding_model, collection_name='exception_resolution_playbook')
playbook_retriever = playbook_vectorstore.as_retriever(search_kwargs={'k': 4})
for d in playbook_retriever.invoke('perishable package weather delay more than 4 hours'):
    print(f"Page {d.metadata['page']} | {d.metadata['section']} | {d.page_content[:90].replace(chr(10), ' ')}...")
