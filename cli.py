# -*- coding: utf-8 -*-
"""命令行入口：一次提问同时运行 RAG 与 CRAG，并输出两者的中间过程与结果。"""
import argparse
import sys

from rag_pipeline import run_rag
from crag_graph import run_crag


def main():
    parser = argparse.ArgumentParser(
        description="RAG / CRAG 对比：输入一个问题，依次执行 RAG 与 CRAG 并输出中间过程与最终答案。"
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        default=None,
        help="要提问的问题（也可在运行后通过提示输入）",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="交互模式：持续输入问题，每次同时跑 RAG 和 CRAG",
    )
    args = parser.parse_args()

    def run_one_question(question: str) -> None:
        question = (question or "").strip()
        if not question:
            print("未输入问题，已跳过。")
            return
        print()
        print("=" * 60)
        print("【RAG 流程与结果】")
        print("=" * 60)
        rag_answer = run_rag(question)
        print("\n[RAG] 最终回答：")
        print(rag_answer)
        print()
        print("=" * 60)
        print("【CRAG 流程与结果】")
        print("=" * 60)
        crag_answer = run_crag(question)
        print("\n[CRAG] 最终回答：")
        print(crag_answer)
        print()

    if args.interactive:
        print("进入交互模式，输入问题后回车运行 RAG + CRAG；输入空行或 Ctrl+C 退出。")
        try:
            while True:
                try:
                    q = input("\n请输入问题: ").strip()
                except EOFError:
                    break
                if not q:
                    break
                run_one_question(q)
        except KeyboardInterrupt:
            print("\n已退出。")
        return

    if args.question:
        run_one_question(args.question)
        return

    # 无 -q 且非 -i 时，从标准输入读一行（便于管道：echo "问题" | python cli.py）
    if not sys.stdin.isatty():
        line = sys.stdin.read().strip()
        run_one_question(line or None)
        return

    # 无参数时提示输入一次
    print("未指定 -q/--question 且未使用 -i。将提示输入一个问题。")
    q = input("请输入问题: ").strip()
    run_one_question(q or None)


if __name__ == "__main__":
    main()
