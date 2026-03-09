import re
import os
import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.settings import Settings

from llama_index.llms.groq import Groq


# =====================================
# LLM (Groq)
# =====================================

groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
if not groq_api_key:
    raise ValueError("Missing GROQ_API_KEY environment variable.")

Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key,
)


# =====================================
# LOAD VECTOR STORE
# =====================================

persist_dir = r"D:\NCKH\agent\NCKH\vector_store"

chroma_client = chromadb.PersistentClient(path=persist_dir)

collection = chroma_client.get_or_create_collection(
    "ctdt_index"
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context
)


# =====================================
# DETECT METADATA
# =====================================

def detect_metadata(question: str):

    program = None
    khoa = None

    q = question.upper()

    if "CNTT" in q:
        program = "CNTT"

    elif "KHMT" in q:
        program = "KHMT"

    elif "NHTTTK" in q:
        program = "NHTTTK"

    match = re.search(r"K(\d+)", q)

    if match:
        khoa = f"K{match.group(1)}"

    return program, khoa


# =====================================
# RAG SEARCH
# =====================================

def rag_search(question, program=None, khoa=None):

    filters = {}

    if program:
        filters["program"] = program

    if khoa:
        filters["khoa"] = khoa

    query_engine = index.as_query_engine(
        similarity_top_k=30,
        filters=filters if filters else None
    )

    response = query_engine.query(question)

    return response


# =====================================
# FALLBACK LLM
# =====================================

def general_llm(question):

    llm = Settings.llm

    result = llm.complete(question)

    return result.text


# =====================================
# TOOL FOR AGENT
# =====================================

def search_ctdt(question: str):

    program, khoa = detect_metadata(question)

    response = rag_search(
        question,
        program=program,
        khoa=khoa
    )

    # nếu RAG không có dữ liệu
    if len(response.source_nodes) == 0:
        return general_llm(question)

    return str(response)
