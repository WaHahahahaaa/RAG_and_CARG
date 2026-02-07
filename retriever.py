# -*- coding: utf-8 -*-
"""向量检索：Embedding 模型与 Chroma 检索器。"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import BGE_MODEL_PATH, VECTOR_DB_DIR, RETRIEVE_TOP_K


def get_embeddings():
    """创建并返回 BGE Embedding 模型。"""
    return HuggingFaceEmbeddings(
        model_name=BGE_MODEL_PATH,
        model_kwargs={"device": "cuda:0"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store():
    """创建并返回 Chroma 向量库。"""
    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=get_embeddings(),
    )


def get_retriever():
    """返回配置好 k 的检索器。"""
    vector_db = get_vector_store()
    return vector_db.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K})
