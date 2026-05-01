import json

from common.neo4j_client import neo4j_client


# === 清洗函数 ===
def clean_text(val):
    if not val or str(val).strip() in ["", "\"\""]:
        return ""
    return str(val).strip()


# === 生成 Cypher 查询 ===
def generate_herb_graph_queries(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        herbs = json.load(f)

    queries = []

    for herb in herbs:
        name = clean_text(herb.get("name"))

        # 创建 Herb 实体
        queries.append((
            "MERGE (h:Herb {name: $name}) SET h.project='TCM'",
            {"name": name}
        ))

        # 处理功效
        for effect in herb.get("effects", []):
            effect_clean = clean_text(effect)
            queries.append((
                "MERGE (e:Effect {name: $name}) SET e.project='TCM'",
                {"name": effect_clean}
            ))
            queries.append((
                "MATCH (h:Herb {name: $herb}), (e:Effect {name: $effect}) "
                "MERGE (h)-[:HAS_EFFECT {project:'TCM'}]->(e)",
                {"herb": name, "effect": effect_clean}
            ))

        # 处理 indications：创建 Disease / Symptom 实体
        for item in herb.get("indications", []):
            content = clean_text(item.get("content"))
            type_ = item.get("type")

            if type_ == "疾病":
                queries.append((
                    "MERGE (d:Disease {name: $name}) SET d.project='TCM'",
                    {"name": content}
                ))
                queries.append((
                    "MATCH (h:Herb {name: $herb}), (d:Disease {name: $disease}) "
                    "MERGE (h)-[:TREATS_DISEASE {project:'TCM'}]->(d)",
                    {"herb": name, "disease": content}
                ))
            elif type_ == "症状":
                queries.append((
                    "MERGE (s:Symptom {name: $name}) SET s.project='TCM'",
                    {"name": content}
                ))
                queries.append((
                    "MATCH (h:Herb {name: $herb}), (s:Symptom {name: $symptom}) "
                    "MERGE (h)-[:ALLEVIATES_SYMPTOM {project:'TCM'}]->(s)",
                    {"herb": name, "symptom": content}
                ))

        # 处理性
        for nature in herb.get("natures", []):
            nature_clean = clean_text(nature)
            queries.append((
                "MERGE (n:HerbNature {name: $name}) SET n.project='TCM'",
                {"name": nature_clean}
            ))
            queries.append((
                "MATCH (h:Herb {name: $herb}), (n:HerbNature {name: $nature}) "
                "MERGE (h)-[:HAS_NATURE {project:'TCM'}]->(n)",
                {"herb": name, "nature": nature_clean}
            ))

        # 处理味
        for flavor in herb.get("flavors", []):
            flavor_clean = clean_text(flavor)
            queries.append((
                "MERGE (f:HerbFlavor {name: $name}) SET f.project='TCM'",
                {"name": flavor_clean}
            ))
            queries.append((
                "MATCH (h:Herb {name: $herb}), (f:HerbFlavor {name: $flavor}) "
                "MERGE (h)-[:HAS_FLAVOR {project:'TCM'}]->(f)",
                {"herb": name, "flavor": flavor_clean}
            ))

        # 处理归经
        for meridian in herb.get("meridians", []):
            meridian_clean = clean_text(meridian)
            queries.append((
                "MERGE (m:Meridian {name: $name}) SET m.project='TCM'",
                {"name": meridian_clean}
            ))
            queries.append((
                "MATCH (h:Herb {name: $herb}), (m:Meridian {name: $meridian}) "
                "MERGE (h)-[:BELONGS_TO_MERIDIAN {project:'TCM'}]->(m)",
                {"herb": name, "meridian": meridian_clean}
            ))

        # 疾病 → 症状 结构关系
        # 安全处理疾病-症状结构关系
        for rel in herb.get("relations") or []:
            disease = clean_text(rel.get("disease"))
            queries.append((
                "MERGE (d:Disease {name: $name}) SET d.project='TCM'",
                {"name": disease}
            ))
            for sym in rel.get("symptoms") or []:
                sym_clean = clean_text(sym)
                queries.append((
                    "MERGE (s:Symptom {name: $name}) SET s.project='TCM'",
                    {"name": sym_clean}
                ))
                queries.append((
                    "MATCH (d:Disease {name: $disease}), (s:Symptom {name: $symptom}) "
                    "MERGE (d)-[:HAS_SYMPTOM {project:'TCM'}]->(s)",
                    {"disease": disease, "symptom": sym_clean}
                ))

    return queries


# === 主程序入口 ===
if __name__ == "__main__":
    json_path = "../__002__extract_node_relation/中药实体关系细节提取结果.json"  # ← 修改为你的 JSON 文件路径

    queries = generate_herb_graph_queries(json_path)

    neo4j_client.run_multiple_cypher(queries)

    print("✅ 中药图谱数据导入完成！")
