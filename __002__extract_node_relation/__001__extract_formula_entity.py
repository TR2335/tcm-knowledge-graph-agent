import os
import pandas as pd
from langchain_core.messages import HumanMessage
from tqdm import tqdm
import json

from common.llm import my_llm

txt_folder_path = "../__001__爬虫部分/方剂"
excel_path = "../__001__爬虫部分/中医方剂列表.xlsx"
intermediate_path = "方剂属性提取结果_temp.json"
final_path = "方剂属性提取结果.xlsx"
start_from_scratch = False

# ============ 字段结构校验函数 ============

'''
txt → AI 提取：名称、别名、出处、组成、功效、主治、用法、禁忌
Excel → 直接读取：功效分类、方剂分类
代码把两部分拼在一起 → 最终标准 Excel
'''
expected_fields = [
    "名称", "别名", "出处", "组成", "功效",
    "主治", "用法", "禁忌", "功效分类", "方剂分类"
]

# 把 AI 返回的文字，拆成 10 个字段
# 输入：AI 返回的文本、期望字段列表
# 输出：字典格式的结构化数据
# 作用：把 AI 返回的文字拆成标准字段。
def parse_structured_output(text: str, expected_fields: list[str]) -> dict:
    # 初始化一个空结果字典，10 个字段先全部设为空字符串。
    result = {field: "" for field in expected_fields}
    # 把 AI 返回的文本去掉首尾空格 + 按行拆分成列表，一行一行处理。
    lines = text.strip().splitlines()
    # 遍历每一行，解析字段值
    # 输出格式是：字段名: 内容（若无内容请填 ""）
    for line in lines:
        # 如果行中没有冒号，跳过该行
        if ":" not in line:
            continue
        # 用第一个冒号把行切分成两部分：
        # k = 字段名（如 名称、功效）
        # v = 字段内容。
        k, v = line.split(":", 1)
        # 清理格式：
        # 字段名 k：去掉空格 + 去掉所有空白（防止 AI 多打空格）
        # 内容 v：只去掉首尾空格。
        k, v = k.strip().replace(" ", ""), v.strip()
        # 遍历我们期望的标准字段列表。
        # expected_fields是方剂的属性字段名列表
        for field in expected_fields:
            # 对比 AI 返回的字段名 和 标准字段名（都去掉空格后对比）。
            if k == field.replace(" ", ""):
                # 如果匹配成功，就把内容填入对应字段，然后跳出循环。
                result[field] = v
                break

    # 匹配成功就把内容填入对应字段，然后跳出循环。
    missing_fields = [f for f in expected_fields if result[f] == ""]
    # 如果缺失字段过多（超过一半），认为输出异常。
    if len(missing_fields) > len(expected_fields) // 2:
        raise ValueError(f"字段缺失过多（共{len(missing_fields)}项）: {missing_fields}")
    return result


# ============ 主处理逻辑 ============

#读取方剂名单 Excel，变成表格数据对象 excel_df。
excel_df = pd.read_excel(excel_path)
results = []
# 创建空集合，用于记录已经处理过的方剂名称（避免重复处理）
processed_names = set()
# 如果不从头开始 + 临时文件存在，就加载之前的结果。
'''
这是通过保存之前的文件intermediate_path在这个地址保存 的文件 
就是方剂属性提取结果_temp.json暂时作为临时文件 
就是之前的结果 在集合里存入这些结果继续循环未处理文件 
然后通过processed_names 过滤掉已经处理过的方剂名称
'''
if not start_from_scratch and os.path.exists(intermediate_path):
    # 打开临时 json 文件，utf-8 编码避免中文乱码。
    with open(intermediate_path, "r", encoding="utf-8") as f:
        results = json.load(f)
        # 把已经处理过的方剂名称存入集合。
        # 集合 里面存入 所有已经处理过的方剂名称。
        processed_names = set(item["名称"] for item in results)
# 遍历 Excel 里的每一行，带进度条显示。
# row = 当前行数据。 iterrows = 生成器，返回每一行的索引和数据。 total = 总行数。
for _, row in tqdm(excel_df.iterrows(), total=len(excel_df)):
    name = row["方剂名称"]
    effect_class = row["功效分类"]
    formula_class = row["方剂大类"]
    # 如果这个方剂已经处理过，直接跳过
    if name in processed_names:
        continue
    # 拼接出该方剂对应的 txt 原文路径。
    txt_path = os.path.join(txt_folder_path, f"{name}.txt")
    if not os.path.exists(txt_path):
        print(f"⚠️ 未找到文件：{txt_path}")
        continue

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 构造 prompt（格式化约束更强）
    prompt = f"""
                你是一位中医方剂知识专家，请从下列方剂资料中提取固定结构的字段内容。
                
                方剂名称：{name}
                资料内容如下（请严格依照原文提取）：
                
                ```
                {content}
                ```
                
                请严格按照如下格式输出结果，字段顺序不能更改，字段名前后禁止添加任何解释或注释。
                
                每一行格式为：字段名: 内容（若无内容请填 ""）
                
                名称: 
                别名: 
                出处: 
                组成: 
                功效: 
                主治: 
                用法: 
                禁忌: 
                功效分类: {effect_class}
                方剂分类: {formula_class}
                    """.strip()

    try:
        response = my_llm.invoke([HumanMessage(content=prompt)])
        # 提取模型输出内容
        output = response.content

        print(f"\n====== {name} 提取结果预览 ======\n{output}\n==========================\n")
        # expected_fields 字段校验 名称
        field_values = parse_structured_output(output, expected_fields)
        results.append(field_values)

        # 实时保存中间结果
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"❌ {name} 提取失败：{e}")

# ============ 最终保存 ============
result_df = pd.DataFrame(results)
result_df.to_excel(final_path, index=False)
print(f"✅ 提取完成，结果保存至 {final_path}")
