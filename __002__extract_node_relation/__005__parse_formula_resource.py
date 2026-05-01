import json
import re


# 特殊处理规则（写死替换）
def fix_special_titles(title):
    special_cases = {
        '《傅青主女科·产后偏上卷》': '《傅青主女科》',
        '《伤寒论：辨太阳病脉证并治中》': '《伤寒论》',
        '《金匮要略●百合狐惑阴阳毒病篇》': '《金匮要略》',
        '《宋·太平惠民和剂局方》': '《太平惠民和剂局方》',
        '《宋•太平惠民和剂局方》': '《太平惠民和剂局方》',
        '《先醒斋医学广笔记·妇人》': '《先醒斋医学广笔记》',
        '《金匮》': '《金匮要略》',
        '《千金》': '《备急千金要方》',
        '《小儿药证直决》': '《小儿药证直诀》',
        '《证治准绳·类方》': '《证治准绳》',
        '《证治准绳·疡医》': '《证治准绳》',
        '《伤寒杂病论》': '《伤寒论》',
        '《伤寒六书》': '《伤寒论》',
        '《校注妇人良方》': '《妇人良方》',
        '《金鉴》': '《医学金鉴》',
        '《千金翼》': '《千金翼方》',
        '《丹溪心法附余》': '《丹溪心法》',
        '《妇人大全良方》': '《妇人良方》',
        '《外科症治全生集》': '《外科全生集》',
        '《卫生宝鉴·补遗》': '《卫生宝鉴》',
        '《伤寒全生集》': '《伤寒论》',
    }
    return special_cases.get(title, title)


# 提取标准书名字段（带书名号）
def extract_standard_title(raw_chuchu):
    match = re.search(r'《[^《》]+》', raw_chuchu)
    if match:
        raw_title = match.group()
        return fix_special_titles(raw_title)
    return None


# 主逻辑：添加“出处标准化”字段并存储
def standardize_and_save(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        chuchu = item.get("出处", "")
        std_title = extract_standard_title(chuchu)
        item["出处标准化"] = std_title if std_title else ""

    # 写回新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 示例调用
if __name__ == "__main__":
    input_file = '方剂属性提取结果_temp.json'  # 输入文件
    output_file = '方剂属性提取结果_temp.json'  # 输出文件
    standardize_and_save(input_file, output_file)
    print(f"已生成标准化文件：{output_file}")
