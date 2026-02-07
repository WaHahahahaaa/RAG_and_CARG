import os
import io
import json
import fitz 
from vllm import LLM
from PIL import Image
from mineru_vl_utils import MinerUClient, MinerULogitsProcessor

from dotenv import load_dotenv

load_dotenv()
# mineru path reminder: ../.cache/modelscope/hub/models/OpenDataLab/MinerU2.5-2509-1.2B
mineru_model_path = os.getenv("MINERU_MODEL_PATH")
pdf_glob_path = os.getenv("PDF_GLOB_PATH")
md_glob_path = os.getenv("MD_GLOB_PATH")

class MinerUParser:
    def __init__(self, model_path, gpu_id="0", gpu_utilization=0.9):
        """
        初始化解析器，加载模型
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.model_path = model_path
        
        print(f"正在初始化模型 (GPU:{gpu_id})... 这可能需要 1-2 分钟")
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_utilization,
            max_model_len=16384,
            logits_processors=[MinerULogitsProcessor]
        )
        self.client = MinerUClient(backend="vllm-engine", vllm_llm=self.llm)
        print("模型初始化完成。")

    def _setup_output_dir(self, pdf_path, output_base_dir):
        """创建输出目录结构"""
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_output_dir = os.path.join(output_base_dir, pdf_name)
        
        dirs = {
            "root": pdf_output_dir,
            "figures": os.path.join(pdf_output_dir, "figures"),
            "tables": os.path.join(pdf_output_dir, "tables"),
            "formulas": os.path.join(pdf_output_dir, "formulas")
        }
        
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
            
        return pdf_name, dirs

    def parse_pdf(self, pdf_path, output_base_dir):
        pdf_name, dirs = self._setup_output_dir(pdf_path, output_base_dir)
        doc = fitz.open(pdf_path)
        
        total_pages = len(doc)
        # 如果 PDF 只有 1 页或 0 页，则不处理（根据原始逻辑，忽略最后一张）
        if total_pages <= 1:
            print(f"文件 {pdf_name} 只有 {total_pages} 页，跳过。")
            return dirs["root"]

        pages_to_process = total_pages - 1
        md_content = []
        counts = {"figure": 0, "table": 0, "formula": 0}
        
        # 标记是否已经识别并跳过了目次页
        found_toc = False

        print(f"开始处理: {pdf_path} (共 {total_pages} 页，跳过最后一页)")

        for page_idx in range(pages_to_process):
            print(f"  正在解析第 {page_idx + 1}/{total_pages} 页...")
            
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=200)
            img_data = pix.tobytes("png")
            page_image = Image.open(io.BytesIO(img_data)).convert("RGB")

            extracted_blocks = self.client.two_step_extract(page_image)

            # --- 新增目次页识别逻辑 ---
            if not found_toc:
                is_this_page_toc = False
                for block in extracted_blocks:
                    b_type = block.get("type", "")
                    content = str(block.get("content", "")).strip()
                    # 判断标准：类型是标题且包含“目次”
                    if b_type == "title" and "目次" in content:
                        print(f"    [检测] 发现目次页（第 {page_idx + 1} 页），已跳过。")
                        is_this_page_toc = True
                        found_toc = True # 标记已找到，后续页面不再检测
                        break
                
                if is_this_page_toc:
                    continue # 跳过当前页的处理
            # -----------------------

            for block in extracted_blocks:
                b_type = block.get("type", "text")
                bbox = block.get("bbox")
                raw_content = block.get("content")
                content = str(raw_content).strip() if raw_content is not None else ""

                if b_type == "title":
                    md_content.append(f"## {content}\n\n")
                elif b_type == "text":
                    if content:
                        md_content.append(f"{content}\n\n")
                elif b_type == "figure":
                    counts["figure"] += 1
                    fname = f"fig_p{page_idx+1}_{counts['figure']}.png"
                    fpath = os.path.join(dirs["figures"], fname)
                    try:
                        page_image.crop(bbox).save(fpath)
                        md_content.append(f"![{fname}](figures/{fname})\n\n")
                    except Exception as e:
                        print(f"    警告: 无法裁剪图片 - {e}")
                elif b_type == "table":
                    counts["table"] += 1
                    tname = f"tab_p{page_idx+1}_{counts['table']}.png"
                    tpath = os.path.join(dirs["tables"], tname)
                    try:
                        page_image.crop(bbox).save(tpath)
                    except: pass
                    with open(os.path.join(dirs["tables"], tname.replace(".png", ".json")), "w") as jf:
                        json.dump(block, jf, ensure_ascii=False)
                    if content:
                        md_content.append(f"{content}\n\n")
                    else:
                        md_content.append(f"![{tname}](tables/{tname})\n\n")
                elif b_type == "formula":
                    if content:
                        if not content.startswith("$"):
                            content = f"$$\n{content}\n$$"
                        md_content.append(f"{content}\n\n")
                        counts["formula"] += 1
                        forname = f"form_p{page_idx+1}_{counts['formula']}.png"
                        try:
                            page_image.crop(bbox).save(os.path.join(dirs["formulas"], forname))
                        except: pass

        # 保存 Markdown
        md_file_path = os.path.join(dirs["root"], f"{pdf_name}.md")
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.writelines(md_content)

        print(f"处理完成！输出文件夹: {dirs['root']}")
        return dirs["root"]


if __name__ == "__main__":
    parser = MinerUParser(
        model_path=mineru_model_path, 
        gpu_id="0"
    )

    import glob
    pdf_files = glob.glob(pdf_glob_path)

    for pdf_file in pdf_files:
        try:
            print(f"--- 正在批量处理: {pdf_file} ---")
            parser.parse_pdf(pdf_path=pdf_file, output_base_dir=md_glob_path)
        except Exception as e:
            print(f"处理文件 {pdf_file} 时出错: {e}")

    print("所有文件处理完毕。")