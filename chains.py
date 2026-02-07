# -*- coding: utf-8 -*-
"""各类 LangChain 链：文档相关性评估、问题重写、最终生成。"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config import (
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_MODEL_NAME,
    LOCAL_LLM_API_KEY,
    DEEPSEEK_NAME,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_API_KEY,
)


# ----- 评估用 LLM（本地） -----
def get_grader_llm():
    return ChatOpenAI(
        model=LOCAL_LLM_MODEL_NAME,
        openai_api_base=LOCAL_LLM_BASE_URL,
        openai_api_key=LOCAL_LLM_API_KEY,
        temperature=0,
    )


GRADER_SYSTEM = """你是一名专业的文档相关性评分员，负责评估检索到的文档与用户提问之间的相关性。
评分标准：
1. 如果文档中包含与用户问题相关的关键词、技术指标、术语定义或语义内容，请将其判定为"yes"（相关）。
2. 只要文档对回答问题有参考价值，即使不能完全回答问题，也应判定为"yes"。
3. 如果文档内容与问题完全无关，请判定为"no"（不相关）。

评价结果仅输出"yes"或"no"，不要输出任何解释或多余的文字。"""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", GRADER_SYSTEM),
    ("human", "检索到的文档内容：\n\n {document} \n\n 用户提问：\n\n {question}"),
])


def get_retrieval_grader():
    """文档相关性评估链：输入 document + question，输出 yes/no。"""
    return grade_prompt | get_grader_llm() | StrOutputParser()


# ----- 生成用 LLM（DeepSeek） -----
def get_generate_llm():
    return ChatOpenAI(
        model=DEEPSEEK_NAME,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0,
    )


GENERATE_TEMPLATE = """你是一名铁路相关知识的专家。请利用以下检索到的上下文来回答问题。
如果你不知道答案，就说你不知道，不要试着编造答案。
最多使用三句话，回答要简明扼要。

上下文: {context}
问题: {question}
答案:"""

generate_prompt = ChatPromptTemplate.from_template(GENERATE_TEMPLATE)


def get_rag_chain():
    """RAG 生成链：输入 context + question，输出答案。"""
    return generate_prompt | get_generate_llm() | StrOutputParser()


# ----- 问题重写用 LLM（DeepSeek） -----
REWRITER_SYSTEM = """你是一名专业的问题改写助手。你的任务是将用户输入的原始问题转化为一个更适合进行网络搜索和文档检索的优化版本。
在改写时，请遵循以下原则：
1. 深入分析问题背后的深层语义意图和核心需求。
2. 提取并使用更专业的术语（尤其是针对铁路信号等技术领域）。
3. 使改写后的问题描述更加清晰、具体，以便搜索引擎或向量数据库能匹配到最高质量的结果。
4. 只输出改写后的问题，不要输出任何解释或多余的文字。"""

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", REWRITER_SYSTEM),
    ("human", "这是原始问题：\n\n {question} \n\n 请给出一个改进后的版本："),
])


def get_question_rewriter():
    """问题重写链：输入 question，输出改写后的问题。"""
    return rewrite_prompt | get_generate_llm() | StrOutputParser()
