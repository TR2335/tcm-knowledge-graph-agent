import json
import os
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

# ========= 1. 定义输出结构 =========

from pydantic import BaseModel, Field
from typing import List, Literal, Optional

from common.llm import my_llm

#  content = 病名或症状名
# str = 必须是文字
# Field 里写说明给 AI 看
# Literal 只能是 疾病 或 症状 二选一
class IndicationItem(BaseModel):
    content: str = Field(..., description="具体的主治内容，如具体的病名或症状名称")
    type: Literal["疾病", "症状"] = Field(..., description="判断是疾病或症状，只能是这两个值之一")

#  herb：药材名称
# amount：药材用量，如 '24克'；如果未提供，则为 None
# Optional 可选参数
# list 和 str 只要是标签 是必须传入值，不能是 None
class IngredientItem(BaseModel):
    herb: str = Field(..., description="药材名称")
    amount: Optional[str] = Field(None, description="药材用量，如 '24克'；如果未提供，则为 None")

#  disease：疾病名称
# symptoms：与该疾病明确相关的症状列表
class DiseaseSymptomRelation(BaseModel):
    disease: str = Field(..., description="疾病名称")
    symptoms: List[str] = Field(..., description="与该疾病明确相关的症状列表")

#  name：中药方剂的名称
#  effects：方剂的功效，用字符串列表表示
#  indications：主治内容，标注出每一项是疾病还是症状
#  ingredients：组成药材及其用量，每项包含药材和分量
#  relations：疾病与症状之间的明确对应关系；如果未明确说明可省略
class FormulaResult(BaseModel):
    name: str = Field(..., description="中药方剂的名称")
    effects: List[str] = Field(..., description="方剂的功效，用字符串列表表示")
    indications: List[IndicationItem] = Field(..., description="主治内容，标注出每一项是疾病还是症状")
    ingredients: List[IngredientItem] = Field(..., description="组成药材及其用量，每项包含药材和分量")
    relations: Optional[List[DiseaseSymptomRelation]] = Field(
        default=None,
        description="疾病与症状之间的明确对应关系；如果未明确说明可省略"
    )

#  root：一个方剂列表，每个为结构化的结果
class FormulaBatch(BaseModel):
    root: List[FormulaResult] = Field(..., description="一个方剂列表，每个为结构化的结果")

#  FormulaBatch 输出解析器
parser = PydanticOutputParser(pydantic_object=FormulaBatch)
# get_format_instructions 获取格式指令
format_instructions = parser.get_format_instructions()


# ========= 3. Prompt 构造函数 =========

def build_structured_prompt(formula_list: List[dict]) -> PromptTemplate:
    prompt_str = "你是资深中医药专家，请对以下中医方剂进行功效与主治的标签化处理：\n\n"
    # 遍历每个方剂，获取名称、组成、功效、主治
    for idx, f in enumerate(formula_list, 1):
        prompt_str += (
            f"方剂{idx}：\n"
            f"名称：{f.get('名称', '')}\n"
            f"组成：{f.get('组成', '')}\n"
            f"功效：{f.get('功效', '')}\n"
            f"主治：{f.get('主治', '')}\n\n"
        )
    prompt_str += "请严格按照以下格式返回结果，是一个 JSON 数组：\n\n"
    prompt_str += "用以下JSON格式返回：\n{format_instructions}\n"
    # 返回  prompt_str 提示词模版内容 输入变量为空 格式指令作为部分变量
    return PromptTemplate(
        template=prompt_str,
        input_variables=[],
        partial_variables={"format_instructions": format_instructions}
    )


# ========= 4. 主处理函数 =========
# 定义批量处理函数：
# data = 要处理的方剂
# output_path = 输出文件
# batch_size = 一次发 5 个给 AI
# resume = 是否断点续跑
def process_batch(data: List[dict], output_path: str, batch_size: int = 5, resume: bool = True):
    # 初始化已处理的名称集合 set() 集合，用于存储已处理的方剂名称
    existing_names = set()
    all_results = []

    # 如果开启续跑 + 文件已存在，就加载之前的结果。
    if resume and os.path.exists(output_path):
        # 读取已保存的结果
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                # 加载已保存的结果
                saved = json.load(f)
                # 遍历已保存的结果，将名称添加到 existing_names 集合中
                for item in saved:
                    # 把已处理的名称加入集合。
                    existing_names.add(item["name"])
                    # 把已处理的结果加入列表。
                    all_results.append(FormulaResult(**item))
                print(f"🔁 已加载 {len(existing_names)} 个已处理方剂，跳过这些内容。")
            except Exception as e:
                print("⚠️ 读取已保存结果失败：")
                # 方法 1：将异常信息解码为字符串再打印（含中文）
                try:
                    error_str = str(e).encode("utf-8").decode("unicode_escape")
                    print("🔍 异常信息（已解码）：", error_str)
                except Exception as decode_error:
                    print("⚠️ 解码异常信息失败，原始异常：", str(e))

    # 过滤出还没处理的方剂。循环 data 列表，筛选出名称不在 existing_names 集合中的方剂。
    data_to_process = [item for item in data if item.get("名称") not in existing_names]
    # 按批次循环，带进度条。
    for i in tqdm(range(0, len(data_to_process), batch_size), desc="处理中..."):
        # 提取当前批次的方剂
        batch = data_to_process[i: i + batch_size]
        # 构建当前批次的提示词
        prompt = build_structured_prompt(batch)
        # 构建当前批次的链 prompt提示词 -> my_llm大模型 -> parser数据格式
        chain = prompt | my_llm | parser

        try:
            # 执行当前批次的链，获取结果
            response: FormulaBatch = chain.invoke({})
            # root 是一个方剂列表，每个为结构化的结果
            results = response.root
            all_results.extend(results)

            # 追加写入文件（断点续存）
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([item.model_dump() for item in all_results], f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"❌ 处理 batch {i // batch_size} 失败：{e}")
            continue

    return all_results


# ========= 5. 主程序入口 =========

if __name__ == "__main__":
    INPUT_FILE = "方剂属性提取结果_temp.json"
    OUTPUT_FILE = "方剂实体关系细节提取结果.json"

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed = process_batch(
        data=raw_data,
        output_path=OUTPUT_FILE,
        batch_size=5,
        resume=True  # ← 可切换成 False 重新开始
    )

    print(f"✅ 处理完成，共处理 {len(processed)} 个方剂。结果保存在：{OUTPUT_FILE}")
'''
把清洗好的方剂 JSON → 发给 AI → 强制输出高精度结构 → 自动拆分：
功效列表、药材 + 用量、主治（疾病 / 症状标注）、疾病 - 症状对应关系 → 实时保存 → 最终输出标准 JSON

'''