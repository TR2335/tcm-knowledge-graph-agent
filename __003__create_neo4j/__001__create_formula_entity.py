import json
from tqdm import tqdm

from common.neo4j_client import neo4j_client


def clean_text(value):
    if not value or value.strip() == "\"\"":
        return ""
    return value.strip()

# 从 JSON 文件加载方剂数据
def load_formulas_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 生成 Neo4j 查询语句
def generate_cypher_queries(formulas):
    queries = []
    # 遍历每个方剂项
    for item in tqdm(formulas,  desc="生成方剂实体"):
        name = clean_text(item.get("名称"))
        alias = clean_text(item.get("别名"))
        source = clean_text(item.get("出处"))
        ingredients = clean_text(item.get("组成"))
        effect = clean_text(item.get("功效"))
        indication = clean_text(item.get("主治"))
        usage = clean_text(item.get("用法"))
        taboo = clean_text(item.get("禁忌"))
        effect_category = clean_text(item.get("功效分类"))
        formula_category = clean_text(item.get("方剂分类"))
        std_source = clean_text(item.get("出处标准化"))

        # 创建 Formula 节点
        queries.append((
            "MERGE (f:Formula {name: $name}) "
            "SET f.alias = $alias, f.source = $source, f.ingredients = $ingredients, "
            "f.effect = $effect, f.indication = $indication, f.usage = $usage, "
            "f.taboo = $taboo, f.effect_category = $effect_category, "
            "f.formula_category = $formula_category, f.std_source = $std_source, "
            "f.project = 'TCM'",
            {
                "name": name,
                "alias": alias,
                "source": source,
                "ingredients": ingredients,
                "effect": effect,
                "indication": indication,
                "usage": usage,
                "taboo": taboo,
                "effect_category": effect_category,
                "formula_category": formula_category,
                "std_source": std_source
            }
        ))

        # 创建分类和来源节点
        if effect_category:
            queries.append((
                "MERGE (ec:EffectCategory {name: $name}) SET ec.project = 'TCM'",
                {"name": effect_category}
            ))

        if formula_category:
            queries.append((
                "MERGE (fc:FormulaCategory {name: $name}) SET fc.project = 'TCM'",
                {"name": formula_category}
            ))

        if std_source:
            queries.append((
                "MERGE (s:Source {name: $name}) SET s.project = 'TCM'",
                {"name": std_source}
            ))

        # 创建关系
        if effect_category:
            queries.append((
                "MATCH (f:Formula {name: $f_name}), (ec:EffectCategory {name: $ec_name}) "
                "MERGE (f)-[:BELONGS_TO_EFFECT_CATEGORY {project: 'TCM'}]->(ec)",
                {"f_name": name, "ec_name": effect_category}
            ))

        if effect_category and formula_category:
            queries.append((
                "MATCH (ec:EffectCategory {name: $ec_name}), (fc:FormulaCategory {name: $fc_name}) "
                "MERGE (ec)-[:BELONGS_TO {project: 'TCM'}]->(fc)",
                {"ec_name": effect_category, "fc_name": formula_category}
            ))

        if std_source:
            queries.append((
                "MATCH (f:Formula {name: $f_name}), (s:Source {name: $s_name}) "
                "MERGE (f)-[:FROM_SOURCE {project: 'TCM'}]->(s)",
                {"f_name": name, "s_name": std_source}
            ))

    return queries


# === 主程序入口 ===
if __name__ == "__main__":
    json_path = "../__002__extract_node_relation/方剂属性提取结果_temp.json"  # ← 替换为你的 JSON 文件路径

    formulas = load_formulas_from_json(json_path)
    queries = generate_cypher_queries(formulas)

    neo4j_client.run_multiple_cypher(queries)

    print("✅ TCM formulas successfully imported into Neo4j.")
