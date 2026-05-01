import json
from common.neo4j_client import neo4j_client


# ===== 工具函数 =====
def clean_text(v):
    if not v or str(v).strip() in ["", "\"\""]:
        return ""
    return str(v).strip()


# ===== Cypher 查询生成器 =====
def generate_formula_graph_queries(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        formulas = json.load(f)

    queries = []

    for formula in formulas:
        name = clean_text(formula.get("name"))
        effects = formula.get("effects", [])
        indications = formula.get("indications", [])
        ingredients = formula.get("ingredients", [])
        relations = formula.get("relations", [])

        # 创建方剂节点
        queries.append((
            "MERGE (f:Formula {name: $name}) SET f.project='TCM'",
            {"name": name}
        ))

        # 创建功效节点及关系
        for effect in effects:
            effect_clean = clean_text(effect)
            queries.append((
                "MERGE (e:Effect {name: $effect}) SET e.project='TCM'",
                {"effect": effect_clean}
            ))
            queries.append((
                "MATCH (f:Formula {name: $name}), (e:Effect {name: $effect}) "
                "MERGE (f)-[:HAS_EFFECT {project:'TCM'}]->(e)",
                {"name": name, "effect": effect_clean}
            ))

        # 创建中药节点及配方关系（含 amount 属性）
        for item in ingredients:
            herb = clean_text(item.get("herb"))
            amount = clean_text(item.get("amount"))
            queries.append((
                "MERGE (h:Herb {name: $herb}) SET h.project='TCM'",
                {"herb": herb}
            ))
            queries.append((
                "MATCH (f:Formula {name: $name}), (h:Herb {name: $herb}) "
                "MERGE (f)-[r:HAS_INGREDIENT {project:'TCM'}]->(h) "
                "SET r.amount = $amount",
                {"name": name, "herb": herb, "amount": amount}
            ))

        # 创建疾病与症状节点及其与方剂的关系
        for item in indications:
            content = clean_text(item.get("content"))
            type_ = item.get("type")
            if type_ == "疾病":
                queries.append((
                    "MERGE (d:Disease {name: $name}) SET d.project='TCM'",
                    {"name": content}
                ))
                queries.append((
                    "MATCH (f:Formula {name: $f_name}), (d:Disease {name: $d_name}) "
                    "MERGE (f)-[:TREATS_DISEASE {project:'TCM'}]->(d)",
                    {"f_name": name, "d_name": content}
                ))
            elif type_ == "症状":
                queries.append((
                    "MERGE (s:Symptom {name: $name}) SET s.project='TCM'",
                    {"name": content}
                ))
                queries.append((
                    "MATCH (f:Formula {name: $f_name}), (s:Symptom {name: $s_name}) "
                    "MERGE (f)-[:ALLEVIATES_SYMPTOM {project:'TCM'}]->(s)",
                    {"f_name": name, "s_name": content}
                ))

        # 疾病与症状关系
        for rel in relations or []:
            disease = clean_text(rel.get("disease"))
            symptom_list = rel.get("symptoms", [])
            queries.append((
                "MERGE (d:Disease {name: $name}) SET d.project='TCM'",
                {"name": disease}
            ))
            for sym in symptom_list:
                symptom_clean = clean_text(sym)
                queries.append((
                    "MERGE (s:Symptom {name: $name}) SET s.project='TCM'",
                    {"name": symptom_clean}
                ))
                queries.append((
                    "MATCH (d:Disease {name: $d_name}), (s:Symptom {name: $s_name}) "
                    "MERGE (d)-[:HAS_SYMPTOM {project:'TCM'}]->(s)",
                    {"d_name": disease, "s_name": symptom_clean}
                ))

    return queries


# ===== 主程序入口 =====
if __name__ == "__main__":
    json_path = "../__002__extract_node_relation/方剂实体关系细节提取结果.json"  # ← 请替换为你的实际路径

    queries = generate_formula_graph_queries(json_path)

    neo4j_client.run_multiple_cypher(queries)

    print("✅ 中医方剂图谱导入完成！")
