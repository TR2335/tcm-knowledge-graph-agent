# -*- coding: utf-8 -*-
import json
import os
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

from common.llm import my_llm


# ===== 1. 数据结构定义 =====
class IndicationItem(BaseModel):
    content: str = Field(..., description="具体的主治内容，如疾病名或症状描述")
    type: Literal["疾病", "症状"] = Field(..., description="标注该主治内容是疾病还是症状")


class IngredientItem(BaseModel):
    herb: str = Field(..., description="药材名称")
    amount: Optional[str] = Field(None, description="药材用量，例如 '3克'；如无则为 None")


class DiseaseSymptomRelation(BaseModel):
    disease: str = Field(..., description="疾病名称")
    symptoms: List[str] = Field(..., description="该疾病明确相关的症状列表")


class HerbResult(BaseModel):
    name: str = Field(..., description="中药名称")
    effects: List[str] = Field(..., description="中药的功效列表")
    indications: List[IndicationItem] = Field(..., description="主治内容列表，并标注是疾病或症状")
    relations: Optional[List[DiseaseSymptomRelation]] = Field(
        default=None,
        description="疾病与症状之间的对应关系（可选）"
    )
    natures: List[Literal["寒", "凉", "温", "热", "平"]] = Field(
        ..., description="中药的性”字段仅允许五种值之一或多种或空：寒、凉、温、热、平。比如“微温”、“大热”一律归类为“温”或“热”"
    )
    flavors: List[Literal["辛", "甘", "酸", "苦", "咸", "淡"]] = Field(
        ..., description="中药的味字段仅允许六种值之一或多种或空：辛、甘、酸、苦、咸、淡"
    )
    meridians: List[
        Literal[
            "肺经", "大肠经", "胃经", "脾经", "心经", "小肠经", "膀胱经",
            "肾经", "心包经", "三焦经", "胆经", "肝经"
        ]
    ] = Field(...,
              description="中药归经，限定为十二正经之一或多种或空，仅允许出现：肺经, 大肠经, 胃经, 脾经, 心经, 小肠经, 膀胱经,肾经, 心包经, 三焦经, 胆经, 肝经")


class HerbBatch(BaseModel):
    root: List[HerbResult] = Field(..., description="结构化提取后的中药列表")


pydantic_parser = PydanticOutputParser(pydantic_object=HerbBatch)
parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=my_llm)

# parser = PydanticOutputParser(pydantic_object=HerbBatch)
format_instructions = parser.get_format_instructions()


# ===== 3. Prompt 构建 =====
def build_herb_prompt(herb_list: List[dict]) -> PromptTemplate:
    prompt_str = ("你是资深中医药专家，请对以下中药材信息进行结构化提取，包括：功效、主治（并区分疾病和症状）、疾病与症状的对应关系、性（寒热属性）、味（六味标准）和归经（十二经）。"
                  "如果某字段缺失，也必须返回空或者空数组（[]），不要省略字段；"
                  "如果disease-symptom 关系 没有symptoms字段，那么就不用展示这个关系\n\n")
    for idx, h in enumerate(herb_list, 1):
        prompt_str += (
            f"中药{idx}：\n"
            f"名称：{h.get('名称', '')}\n"
            f"功效：{h.get('功效', '')}\n"
            f"主治：{h.get('主治', '')}\n"
            f"性味：{h.get('性味', '')}\n"
            f"归经：{h.get('归经', '')}\n\n"
        )
    prompt_str += "请严格按照以下格式返回结果，是一个 JSON 数组：\n"
    prompt_str += f"\n格式要求如下：\n{{format_instructions}}\n"
    return PromptTemplate(
        template=prompt_str,
        input_variables=[],
        partial_variables={"format_instructions": format_instructions}
    )


# ===== 4. 批量处理函数 =====
def process_herb_batch(data: List[dict], output_path: str, batch_size: int = 5, resume: bool = True):
    existing_names = set()
    all_results = []

    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                saved = json.load(f)
                for item in saved:
                    existing_names.add(item["name"])
                    all_results.append(HerbResult(**item))
                print(f"🔁 已加载 {len(existing_names)} 个已处理中药，跳过这些内容。")
            except Exception as e:
                print("⚠️ 读取已保存结果失败：")
                # 方法 1：将异常信息解码为字符串再打印（含中文）
                try:
                    error_str = str(e).encode("utf-8").decode("unicode_escape")
                    print("🔍 异常信息（已解码）：", error_str)
                except Exception as decode_error:
                    print("⚠️ 解码异常信息失败，原始异常：", str(e))

    data_to_process = [item for item in data if item.get("名称") not in existing_names]

    for i in tqdm(range(0, len(data_to_process), batch_size), desc="处理中药数据中..."):
        batch = data_to_process[i: i + batch_size]
        prompt = build_herb_prompt(batch)
        chain = prompt | llm | parser

        try:
            response: HerbBatch = chain.invoke({})
            results = response.root
            all_results.extend(results)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([item.model_dump() for item in all_results], f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"❌ 处理 batch {i // batch_size} 失败：{e}")
            continue

    return all_results


# ===== 5. 主程序入口 =====
if __name__ == "__main__":
    INPUT_FILE = "中药属性提取结果_temp.json"
    OUTPUT_FILE = "中药实体关系细节提取结果.json"

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed = process_herb_batch(
        data=raw_data,
        output_path=OUTPUT_FILE,
        batch_size=5,
        resume=True
    )

    print(f"✅ 处理完成，共处理 {len(processed)} 个中药。结果保存在：{OUTPUT_FILE}")
