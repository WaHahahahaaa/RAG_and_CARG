# -*- coding: utf-8 -*-
"""项目配置：从环境变量加载并暴露给各模块使用。"""
import os
from dotenv import load_dotenv

load_dotenv()

# Embedding 与向量库
BGE_MODEL_PATH = os.getenv("BGE_MODEL_PATH")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR")

# 评估用本地模型（文档相关性打分）
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL")
LOCAL_LLM_MODEL_NAME = os.getenv("LOCAL_LLM_MODEL_NAME")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY")

# 生成用模型（DeepSeek）
DEEPSEEK_NAME = os.getenv("DEEPSEEK_NAME")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Tavily 联网搜索
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 检索数量
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "3"))

# 联网搜索返回条数
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))
