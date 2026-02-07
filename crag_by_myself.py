import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import langchainhub as hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from pprint import pprint
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch

load_dotenv()
# embedding+向量数据库
bge_path = os.getenv("BGE_MODEL_PATH")
vector_db_path = os.getenv("VECTOR_DB_DIR")

# 评估模型
api_grader_base = os.getenv("LOCAL_LLM_BASE_URL")
grader_name = os.getenv("LOCAL_LLM_MODEL_NAME")
grader_key = os.getenv("LOCAL_LLM_API_KEY")

# 生成模型
deepseek_name = os.getenv("DEEPSEEK_NAME")
deepseek_url = os.getenv("DEEPSEEK_BASE_URL")
deepseek_key = os.getenv("DEEPSEEK_API_KEY")

# tavily
tavily_api_key = os.getenv("TAVILY_API_KEY")

# ----------------------------------------
# 加载数据库
# ----------------------------------------

# export HF_ENDPOINT=https://hf-mirror.com
embeddings = HuggingFaceEmbeddings(
    model_name=bge_path,
    model_kwargs={'device': 'cuda:0'},
    encode_kwargs={'normalize_embeddings': True}
)

vector_db = Chroma(
    persist_directory=vector_db_path,
    embedding_function=embeddings
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})


# ----------------------------------------
# 检索评估器
# ----------------------------------------

class GradeDocuments(BaseModel):
    """对检索到的文档进行相关性检查的二元评分。"""

    binary_score: str = Field(
        description="文档是否与问题相关，'yes'（相关）或 'no'（不相关）"
    )

# LLM 
grader_llm = ChatOpenAI(
        model=grader_name,
        openai_api_base=api_grader_base,
        openai_api_key=grader_key,
        temperature=0
    )

# Prompt
system = """你是一名专业的文档相关性评分员，负责评估检索到的文档与用户提问之间的相关性。
评分标准：
1. 如果文档中包含与用户问题相关的关键词、技术指标、术语定义或语义内容，请将其判定为“yes”（相关）。
2. 只要文档对回答问题有参考价值，即使不能完全回答问题，也应判定为“yes”。
3. 如果文档内容与问题完全无关，请判定为“no”（不相关）。

评价结果仅输出“yes”或“no”，不要输出任何解释或多余的文字。"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "检索到的文档内容：\n\n {document} \n\n 用户提问：\n\n {question}"),
    ]
)

retrieval_grader = grade_prompt | grader_llm | StrOutputParser()

# ----------------------------------------
# 生成
# ----------------------------------------

# Prompt
prompt = ChatPromptTemplate.from_template(
    """你是一名铁路相关知识的专家。请利用以下检索到的上下文来回答问题。
    如果你不知道答案，就说你不知道，不要试着编造答案。
    最多使用三句话，回答要简明扼要。

    上下文: {context}
    问题: {question}
    答案:""")

# LLM
generate_llm = ChatOpenAI(
    model_name=deepseek_name, 
    openai_api_key=deepseek_key,
    openai_api_base=deepseek_url,
    temperature=0
)

rag_chain = prompt | generate_llm | StrOutputParser()



# ----------------------------------------
#  问题重写器
# ----------------------------------------

# LLM 
rewriter_llm = ChatOpenAI(
    model_name=deepseek_name,
    openai_api_key=deepseek_key,
    openai_api_base=deepseek_url,
    temperature=0
)

# Prompt
system = """你是一名专业的问题改写助手。你的任务是将用户输入的原始问题转化为一个更适合进行网络搜索和文档检索的优化版本。
在改写时，请遵循以下原则：
1. 深入分析问题背后的深层语义意图和核心需求。
2. 提取并使用更专业的术语（尤其是针对铁路信号等技术领域）。
3. 使改写后的问题描述更加清晰、具体，以便搜索引擎或向量数据库能匹配到最高质量的结果。
4. 只输出改写后的问题，不要输出任何解释或多余的文字。"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "这是原始问题：\n\n {question} \n\n 请给出一个改进后的版本：",
        ),
    ]
)
question_rewriter = re_write_prompt | rewriter_llm | StrOutputParser()



# ----------------------------------------
#  Web Search Tool: tavily
# ----------------------------------------
web_search_tool = TavilySearch(max_results=3)





class GraphState(TypedDict):
    """
    表示图的状态。

    属性:
        question: 用户提出的问题
        generation: LLM 生成的最终答案
        web_search: 决策标识，是否需要进行网页搜索
        documents: 检索到的文档列表（包含本地和网页检索的结果）
    """
    question: str
    generation: str
    web_search: str
    documents: List[Document]

### 定义节点
def retrieve(state):
    """
    从向量数据库检索文档。
    """
    print("---执行：检索本地数据库---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state):
    """
    评估检索到的文档与问题的相关性。
    """
    print("---执行：评估文档相关性---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = "No"  # 默认不需要网页搜索

    for d in documents:
        # 使用之前定义的 retrieval_grader
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score if hasattr(score, 'binary_score') else score
        
        if grade.lower() == "yes":
            print("---评分：文档相关---")
            filtered_docs.append(d)
        else:
            print("---评分：文档不相关---")
            web_search = "Yes"  # 只要有一个不相关，就标记需要网页搜索
            continue
            
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def transform_query(state):
    """
    改写问题，以获得更好的检索效果。
    """
    print("---执行：重写搜索查询---")
    question = state["question"]
    documents = state["documents"]

    # 使用之前定义的 question_rewriter
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state):
    """
    基于改写后的问题进行联网搜索。
    """
    print("---执行：联网搜索---")
    question = state["question"]
    documents = state["documents"]

    # 调用 Tavily 搜索
    docs = web_search_tool.invoke({"query": question})

    # TavilySearch 返回 list[dict] 或 带 "results" 的 dict，统一成 list
    results = docs.get("results", docs) if isinstance(docs, dict) else docs
    content = "\n".join([r["content"] for r in results])
    web_results = Document(page_content=content)
    documents.append(web_results)

    return {"documents": documents, "question": question}


def generate(state):
    """
    生成最终回答。
    """
    print("---执行：生成最终回答---")
    question = state["question"]
    documents = state["documents"]

    # 运行 RAG 链
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

### 定义边
def decide_to_generate(state):
    """
    决定是直接生成回答，还是先进行查询转换。
    """
    print("---决策：评估检索结果---")
    web_search = state["web_search"]

    if web_search == "Yes":
        # 发现不相关文档，需要重写问题并联网
        print("---决策结果：部分文档不相关，执行查询转换和联网搜索---")
        return "transform_query"
    else:
        # 文档全部相关，直接生成
        print("---决策结果：文档质量良好，直接生成回答---")
        return "generate"




workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)  
workflow.add_node("grade_documents", grade_documents)  
workflow.add_node("generate", generate)  
workflow.add_node("transform_query", transform_query)  
workflow.add_node("web_search_node", web_search)  

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()




inputs = {"question": "请问一下铁路交通的发展现状"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")
pprint(value["generation"])