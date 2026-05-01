from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
from tqdm import tqdm
import json
from common.config import Config
from common.llm import my_llm
from langchain_core.messages import HumanMessage

conf = Config()


class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver:
            self.driver.close()

    def check_cypher_syntax(self, query, parameters=None):
        explain_query = f"EXPLAIN {query}"
        try:
            with self.driver.session() as session:
                session.run(explain_query, parameters or {})
            return True, "语法正确"
        except CypherSyntaxError as e:
            return False, f"语法错误: {str(e)}"
        except Exception as e:
            return False, f"其他错误: {str(e)}"

    def run_cypher(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def _fix_cypher_with_llm(self, queries_with_params, error_msg):
        queries_text = json.dumps(queries_with_params, ensure_ascii=False, indent=2)
        fix_prompt = f"""你是一个 Neo4j Cypher 专家。以下是一条批量 Cypher 执行失败时的错误信息：
{error_msg}
原始查询列表（共 {len(queries_with_params)} 条）：
{queries_text}
请分析错误原因，并修复这些 Cypher 查询。只返回 JSON 数组，不要有任何解释文字。"""
        try:
            response = my_llm.invoke([HumanMessage(content=fix_prompt)])
            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            fixed = json.loads(content)
            return fixed
        except Exception as e:
            return queries_with_params

    def run_multiple_cypher(self, queries_with_params, max_retries=3):
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                with self.driver.session() as session:
                    def transaction_logic(tx):
                        for query, params in tqdm(queries_with_params, desc="执行 Cypher 语句"):
                            tx.run(query, params or {})
                    session.execute_write(transaction_logic)
                    return
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    self._fix_cypher_with_llm(queries_with_params, str(e))
                else:
                    raise RuntimeError(f"Neo4j 事务执行失败，已重试 {max_retries} 次仍失败。") from last_error

    def get_tcm_node_labels(self):
        query = """MATCH (n) WHERE n.project = "TCM" UNWIND labels(n) AS label RETURN DISTINCT label"""
        with self.driver.session() as session:
            result = session.run(query)
            return [record["label"] for record in result]

    def get_tcm_relationship_structure(self):
        query = """
        MATCH (n)-[r]->(m) WHERE n.project = "TCM"
        WITH head(labels(n)) AS from_label, type(r) AS rel_type, head(labels(m)) AS to_label
        RETURN DISTINCT from_label, rel_type, to_label
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

    def export_tcm_metadata_to_json(self, output_path="tcm_metadata.json"):
        with self.driver.session() as session:
            label_query = 'MATCH (n) WHERE n.project = "TCM" UNWIND labels(n) AS label RETURN DISTINCT label'
            labels = [record["label"] for record in session.run(label_query)]
            rel_query = 'MATCH (n)-[r]-() WHERE n.project = "TCM" RETURN DISTINCT type(r) AS rel_type'
            rel_types = [record["rel_type"] for record in session.run(rel_query)]
            triple_query = """
            MATCH (n)-[r]->(m) WHERE n.project = "TCM"
            WITH head(labels(n)) AS from_label, type(r) AS rel_type, head(labels(m)) AS to_label
            RETURN DISTINCT from_label, rel_type, to_label
            """
            triples = [{"from": r["from_label"], "rel_type": r["rel_type"], "to": r["to_label"], "description": ""}
                       for r in session.run(triple_query)]
            node_props_query = """
            MATCH (n) WHERE n.project = "TCM" UNWIND labels(n) AS label UNWIND keys(n) AS prop
            RETURN DISTINCT label, prop ORDER BY label, prop
            """
            label_props = {}
            for record in session.run(node_props_query):
                label = record["label"]
                prop = record["prop"]
                if prop == "project":
                    continue
                label_props.setdefault(label, []).append({"name": prop, "description": ""})
            rel_props_query = """
            MATCH (n)-[r]->(m) WHERE n.project = "TCM" UNWIND keys(r) AS prop
            RETURN DISTINCT type(r) AS rel_type, prop ORDER BY rel_type, prop
            """
            rel_type_props = {}
            for record in session.run(rel_props_query):
                rel_type = record["rel_type"]
                prop = record["prop"]
                rel_type_props.setdefault(rel_type, []).append({"name": prop, "description": ""})
            json_obj = {
                "labels": [{"name": label, "description": "", "properties": label_props.get(label, [])} for label in labels],
                "relationships": [{"type": rel, "description": "", "properties": rel_type_props.get(rel, [])} for rel in rel_types],
                "triples": triples
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, ensure_ascii=False, indent=2)
            return output_path

    def get_all_disease_and_symptom_names(self):
        query = "MATCH (n) WHERE n:Symptom OR n:Disease RETURN n.name AS name, labels(n)[0] AS label"
        with self.driver.session() as session:
            result = session.run(query)
            return [(record["name"], record["label"]) for record in result]

    def get_all_effect_names(self):
        query = "MATCH (n) WHERE n:Effect RETURN n.name AS name, labels(n)[0] AS label"
        with self.driver.session() as session:
            result = session.run(query)
            return [(record["name"], record["label"]) for record in result]

    def __del__(self):
        self.close()


neo4j_client = Neo4jClient(conf.NEO4J_URI, conf.NEO4J_USER, conf.NEO4J_PASSWORD)
