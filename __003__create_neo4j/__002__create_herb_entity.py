import json

from common.neo4j_client import neo4j_client


def clean_text(value):
    if not value or str(value).strip() in ["", "\"\""]:
        return ""
    return str(value).strip()


def generate_herb_queries(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        herbs = json.load(f)

    queries = []
    herb_names = set()
    herb_alias_map = {}

    # 第一轮：创建节点和一方关系
    for herb in herbs:
        name = clean_text(herb.get("名称"))
        if not name:
            continue

        alias = clean_text(herb.get("别名"))
        origin = clean_text(herb.get("来源"))
        place = clean_text(herb.get("产地"))
        property_flavor = clean_text(herb.get("性味"))
        meridian = clean_text(herb.get("归经"))
        processing = clean_text(herb.get("炮制"))
        traits = clean_text(herb.get("性状"))
        effect = clean_text(herb.get("功效"))
        indication = clean_text(herb.get("主治"))
        dosage = clean_text(herb.get("用法用量"))
        taboo = clean_text(herb.get("禁忌"))
        effect_category = clean_text(herb.get("功效分类"))
        herb_category = clean_text(herb.get("中药分类"))

        # 记录中药名称和别名列表
        herb_names.add(name)
        herb_alias_map[name] = [clean_text(a) for a in alias.split("、") if clean_text(a) and clean_text(a) != name]

        # 创建中药节点
        queries.append((
            "MERGE (h:Herb {name: $name}) "
            "SET h.alias=$alias, h.origin=$origin, h.place=$place, h.property_flavor=$property_flavor, "
            "h.meridian=$meridian, h.processing=$processing, h.traits=$traits, h.effect=$effect, "
            "h.indication=$indication, h.dosage=$dosage, h.taboo=$taboo, "
            "h.effect_category=$effect_category, h.herb_category=$herb_category, h.project='TCM'",
            {
                "name": name, "alias": alias, "origin": origin, "place": place,
                "property_flavor": property_flavor, "meridian": meridian, "processing": processing,
                "traits": traits, "effect": effect, "indication": indication,
                "dosage": dosage, "taboo": taboo,
                "effect_category": effect_category, "herb_category": herb_category
            }
        ))

        # 建立功效分类节点
        if effect_category:
            queries.append((
                "MERGE (ec:EffectCategory {name: $name}) SET ec.project='TCM'",
                {"name": effect_category}
            ))

        # 建立中药分类节点
        if herb_category:
            queries.append((
                "MERGE (hc:HerbCategory {name: $name}) SET hc.project='TCM'",
                {"name": herb_category}
            ))

        # 建立关系：Herb → EffectCategory
        if effect_category:
            queries.append((
                "MATCH (h:Herb {name: $h_name}), (ec:EffectCategory {name: $ec_name}) "
                "MERGE (h)-[:BELONGS_TO_EFFECT_CATEGORY {project:'TCM'}]->(ec)",
                {"h_name": name, "ec_name": effect_category}
            ))

        # 建立关系：EffectCategory → HerbCategory
        if effect_category and herb_category:
            queries.append((
                "MATCH (ec:EffectCategory {name: $ec_name}), (hc:HerbCategory {name: $hc_name}) "
                "MERGE (ec)-[:BELONGS_TO_HERB_CATEGORY {project:'TCM'}]->(hc)",
                {"ec_name": effect_category, "hc_name": herb_category}
            ))

    # 第二轮：添加中药之间的双向别名关系
    for name, aliases in herb_alias_map.items():
        for alias in aliases:
            if alias in herb_names:
                # 正向
                queries.append((
                    "MATCH (h1:Herb {name: $name1}), (h2:Herb {name: $name2}) "
                    "MERGE (h1)-[:ALIAS_OF {project:'TCM'}]->(h2)",
                    {"name1": name, "name2": alias}
                ))
                # 反向
                queries.append((
                    "MATCH (h1:Herb {name: $name1}), (h2:Herb {name: $name2}) "
                    "MERGE (h2)-[:ALIAS_OF {project:'TCM'}]->(h1)",
                    {"name1": name, "name2": alias}
                ))

    return queries


# === 主程序入口 ===
if __name__ == "__main__":
    json_path = "../__002__extract_node_relation/中药属性提取结果_temp.json"  # ← 请替换为你的本地 JSON 路径

    queries = generate_herb_queries(json_path)

    neo4j_client.run_multiple_cypher(queries)

    print("✅ 中药知识图谱已成功导入 Neo4j。")
