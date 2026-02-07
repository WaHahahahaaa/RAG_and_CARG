# -*- coding: utf-8 -*-
"""图状态与评估输出等数据结构。"""
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    CRAG 图的状态。
    属性:
        question: 用户问题
        generation: 最终生成的答案
        web_search: 是否需要进行联网搜索（Yes/No）
        documents: 当前文档列表（本地检索 + 可选联网结果）
    """
    question: str
    generation: str
    web_search: str
    documents: List[Document]
