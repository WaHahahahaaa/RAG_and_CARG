# -*- coding: utf-8 -*-
"""普通 RAG 流程：检索 -> 生成，并输出中间过程。"""
from langchain_core.documents import Document

from retriever import get_retriever
from chains import get_rag_chain


def _format_context(documents):
    """将文档列表转为上下文字符串。"""
    if not documents:
        return ""
    return "\n\n".join(doc.page_content for doc in documents)


def run_rag(question: str) -> str:
    """
    执行普通 RAG：检索本地数据库 -> 生成回答。
    会打印中间步骤，并返回最终答案。
    """
    print("---[RAG] 执行：检索本地数据库---")
    retriever = get_retriever()
    documents = retriever.invoke(question)
    print(f"---[RAG] 检索到 {len(documents)} 条文档---")

    print("---[RAG] 执行：生成最终回答---")
    rag_chain = get_rag_chain()
    context = _format_context(documents)
    generation = rag_chain.invoke({"context": context, "question": question})
    return generation
