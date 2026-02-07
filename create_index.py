import os
import glob
from typing import List
from tqdm import tqdm

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class CRAGDataProcessor:
    def __init__(self, 
                 db_path: str = "./chroma_db", 
                 model_name: str = "BAAI/bge-small-zh-v1.5",
                 device: str = "cpu"):
        """
        初始化处理器
        :param db_path: ChromaDB 存储路径
        :param model_name: BGE 模型路径或名称 (例如 BAAI/bge-large-zh-v1.5)
        :param device: 运行设备 'cpu' 或 'cuda'
        """
        self.db_path = db_path
        self.model_name = model_name
        self.device = device
        
        # 1. 初始化 BGE Embedding
        print(f"正在加载 BGE 模型: {model_name}...")
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}  
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
    def split_markdown(self, md_files: List[str]) -> List[Document]:
        """
        按照层级和长度切分 Markdown 文档
        """
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, 
            strip_headers=False
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, 
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "；", " ", ""]
        )

        all_chunks = []
        print(f"开始切分 {len(md_files)} 个文档...")

        for file_path in tqdm(md_files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # 按标题切分
                header_splits = header_splitter.split_text(content)

                for split in header_splits:
                    metadata = split.metadata
                    source_name = os.path.basename(file_path)
                    
                    # 二级切分
                    sub_docs = text_splitter.split_documents([split])
                    
                    for doc in sub_docs:
                        # 注入元数据
                        path_context = " > ".join([v for k, v in metadata.items() if k.startswith("Header")])
                        doc.metadata["source"] = source_name
                        doc.metadata["full_path"] = path_context
                        all_chunks.append(doc)
            except Exception as e:
                print(f"处理文件 {file_path} 出错: {e}")

        return all_chunks

    def store_in_chroma(self, chunks: List[Document]):
        """
        将切分后的块存入 ChromaDB
        """
        print(f"正在存入 ChromaDB (路径: {self.db_path})... 总块数: {len(chunks)}")
        
        # 如果数据库已存在，它会加载并添加数据；如果不存在则新建
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        # persist() 在新版本 LangChain 中通常会自动调用，手动调用确保安全
        # vectorstore.persist()
        print("数据存储完成！")
        return vectorstore

    def run_pipeline(self, input_dir: str):
        """
        一键运行完整流程
        """
        # 获取所有 md 文件
        md_files = glob.glob(os.path.join(input_dir, "**/*.md"), recursive=True)
        if not md_files:
            print("未找到 .md 文件，请检查路径。")
            return
        print(f"共找到{len(md_files)}个.md文件")
        # 1. 切分
        chunks = self.split_markdown(md_files)
        
        # 2. 存储 (Embedding 在此处内部调用)
        vectorstore = self.store_in_chroma(chunks)
        
        return vectorstore

# --- 使用示例 ---
if __name__ == "__main__":
    # 配置参数
    INPUT_ROOT = "/home/bigdata15/ztt/mineru_OCRpdf/output"  # 你的 md 文件所在目录
    CHROMA_DIR = "./vector_db/railway_db"
    BGE_MODEL = "BAAI/bge-small-zh-v1.5" # 也可以换成本地绝对路径
    
    # 实例化并运行
    processor = CRAGDataProcessor(
        db_path=CHROMA_DIR, 
        model_name=BGE_MODEL,
        device="cuda:0" # 如果有 GPU 可以改为 "cuda"
    )
    
    # 运行处理流程
    vector_db = processor.run_pipeline(INPUT_ROOT)
    
    # 测试检索
    query = "产品出厂前，包装上的合格证应哪些内容？"
    docs = vector_db.similarity_search(query, k=3)
    
    print("\n--- 检索测试 ---")
    for i, doc in enumerate(docs):
        print(f"结果 {i+1} 来自 [{doc.metadata['source']}] ({doc.metadata['full_path']}):")
        print(f"内容摘要: {doc.page_content}\n")