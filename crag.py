# -*- coding: utf-8 -*-
"""
CRAG 单流程运行脚本（仅 CRAG，用于与原有行为一致）。
推荐使用命令行对比 RAG 与 CRAG：
  python cli.py -q "你的问题"
  python cli.py -i   # 交互模式
"""
from crag_graph import run_crag

if __name__ == "__main__":
    question = "中国铁路的发展现状，有哪些严格规定"
    print("---CRAG 运行（默认问题）---\n")
    result = run_crag(question)
    print("\n---最终回答---")
    print(result)
