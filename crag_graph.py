# -*- coding: utf-8 -*-
"""CRAG 图：检索 -> 评估文档 -> 决策 ->（可选）查询重写与联网搜索 -> 生成。"""
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START
from langchain_tavily import TavilySearch

from config import WEB_SEARCH_MAX_RESULTS
from models import GraphState
from retriever import get_retriever
from chains import (
    get_retrieval_grader,
    get_rag_chain,
    get_question_rewriter,
)


def _format_context(documents):
    if not documents:
        return ""
    return "\n\n".join(doc.page_content for doc in documents)


# ---------- 节点 ----------
def retrieve(state: GraphState) -> dict:
    """从向量数据库检索文档。"""
    print("---执行：检索本地数据库---")
    question = state["question"]
    retriever = get_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state: GraphState) -> dict:
    """评估检索到的文档与问题的相关性。"""
    print("---执行：评估文档相关性---")
    question = state["question"]
    documents = state["documents"]
    grader = get_retrieval_grader()

    filtered_docs = []
    web_search = "No"

    for d in documents:
        score = grader.invoke({"question": question, "document": d.page_content})
        grade = getattr(score, "binary_score", score) if hasattr(score, "binary_score") else score
        if isinstance(grade, str) and grade.strip().lower() == "yes":
            print("---评分：文档相关---")
            filtered_docs.append(d)
        else:
            print("---评分：文档不相关---")
            web_search = "Yes"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def transform_query(state: GraphState) -> dict:
    """重写问题以便更好地检索/搜索。"""
    print("---执行：重写搜索查询---")
    question = state["question"]
    documents = state["documents"]
    rewriter = get_question_rewriter()
    better_question = rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state: GraphState) -> dict:
    """使用改写后的问题进行联网搜索，结果并入 documents。"""
    print("---执行：联网搜索---")
    question = state["question"]
    documents = list(state["documents"])
    tool = TavilySearch(max_results=WEB_SEARCH_MAX_RESULTS)
    docs = tool.invoke({"query": question})
    results = docs.get("results", docs) if isinstance(docs, dict) else docs
    content = "\n".join([r["content"] for r in results])
    documents.append(Document(page_content=content))
    return {"documents": documents, "question": question}


def generate(state: GraphState) -> dict:
    """根据当前文档和问题生成最终回答。"""
    print("---执行：生成最终回答---")
    question = state["question"]
    documents = state["documents"]
    rag_chain = get_rag_chain()
    context = _format_context(documents)
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def decide_to_generate(state: GraphState) -> str:
    """根据评估结果决定是直接生成还是先重写+联网。"""
    print("---决策：评估检索结果---")
    if state["web_search"] == "Yes":
        print("---决策结果：部分文档不相关，执行查询转换和联网搜索---")
        return "transform_query"
    print("---决策结果：文档质量良好，直接生成回答---")
    return "generate"


# ---------- 构建图 ----------
def build_crag_app():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_crag(question: str) -> str:
    """
    运行 CRAG 图，打印中间过程，返回最终答案。
    """
    app = build_crag_app()
    inputs = {"question": question, "documents": [], "generation": "", "web_search": "No"}
    final_state = None
    for output in app.stream(inputs):
        for key in output:
            print(f'"Node \'{key}\':"')
        final_state = output
    if not final_state:
        return ""
    # 最后一个节点是 generate，输出里带 generation
    for node_name, node_out in final_state.items():
        if isinstance(node_out, dict) and "generation" in node_out:
            return node_out["generation"]
    return ""
