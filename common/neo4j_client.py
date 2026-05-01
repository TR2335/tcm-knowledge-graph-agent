from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
from tqdm import tqdm
import json
from common.config import Config
from common.llm import my_llm
from langchain_core.messages import HumanMessage

conf = Config()

# 存入数据库
class Neo4jClient:
    def __init__(self, uri, user, password):
        """初始化 Neo4j 驱动"""
        # 初始化方法
        # 传入地址、账号、密码
        # 创建数据库连接驱动，保持连接
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()

    def check_cypher_syntax(self, query, parameters=None):
        """使用 EXPLAIN explain检查 Cypher 语法是否正确"""
        # 检查 Cypher 语法是否正确
        explain_query = f"EXPLAIN {query}"
        try:
            # 执行 EXPLAIN 查询，检查语法是否正确
            # self.driver.session()是执行下文
            # 是 连接数据库的会话对象
            with self.driver.session() as session:
                session.run(explain_query, parameters or {})
            return True, "语法正确"
        except CypherSyntaxError as e:
            return False, f"语法错误: {str(e)}"
        except Exception as e:
            return False, f"其他错误: {str(e)}"

    # 单条查询 方法
    def run_cypher(self, query, parameters=None):
        # 创建会话对象，用于执行查询
        with self.driver.session() as session:
            # 查询 query 并返回结果
            # parameters 就是一个字典，用来安全地把值传给 Cypher。
            result = session.run(query, parameters or {})
            # 转换为列表，每个元素是一个字典
            return [record.data() for record in result]

    '''
    params = 参数化查询  相当于填空查询 query语句 查询 的答案直接填入 params
    代码与数据分离，用户输入永远不会被当成代码执行
    防注入、防破坏、最安全的数据库操作方式
    '''

    def _fix_cypher_with_llm(self, queries_with_params, error_msg):
        """
        使用 LLM 自动修复失败的 Cypher 查询
        :param queries_with_params: List[(query, params)] 原始查询列表
        :param error_msg: str 错误信息
        :return: List[(query, params)] 修复后的查询列表
        """
        # 构造修复 prompt
        queries_text = json.dumps(queries_with_params, ensure_ascii=False, indent=2)

        fix_prompt = f"""你是一个 Neo4j Cypher 专家。以下是一条批量 Cypher 执行失败时的错误信息：

错误信息：
{error_msg}

原始查询列表（共 {len(queries_with_params)} 条）：
{queries_text}

请分析错误原因，并修复这些 Cypher 查询。修复要求：
1. 只修改有问题的部分，保持正确的部分不变
2. 常见的 Neo4j 语法错误包括：
   - 属性名使用了保留字（如 name, type 等需要用反引号 `name`）
   - 标签名或关系类型名包含非法字符
   - 参数传递格式错误
   - 空格或标点符号问题
3. 修复后返回完整的查询列表，格式与输入完全一致
4. 只返回 JSON 数组，不要有任何解释文字

请直接返回修复后的 JSON：
"""
        try:
            response = my_llm.invoke([HumanMessage(content=fix_prompt)])
            # 尝试从返回内容中提取 JSON
            content = response.content.strip()
            # 处理可能的 markdown 代码块
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])  # 去掉 ```json 和 ```
            fixed = json.loads(content)
            print(f"✅ LLM 成功修复了查询，共 {len(fixed)} 条")
            return fixed
        except Exception as e:
            print(f"⚠️ LLM 修复失败: {e}")
            return queries_with_params  # 修复失败时返回原始查询

    # 多条查询 方法 用了回滚事务，支持 LLM 自动修复
    def run_multiple_cypher(self, queries_with_params, max_retries=3):
        """
        执行多条 Cypher 语句，支持自动重试 + LLM 修复
        :param queries_with_params: List[(query, params)] 要执行的 Cypher 语句列表
        :param max_retries: int 最大重试次数，默认3次

        数据流程：
        1. 接收传入的 queries_with_params
        2. 执行，失败则调用 LLM 修复（直接修改传入的列表）
        3. 用修复后的列表重试，循环直到成功或达到最大重试次数
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                with self.driver.session() as session:
                    def transaction_logic(tx):
                        # 直接使用传入的 queries_with_params，失败时被 LLM 修复后更新
                        for query, params in tqdm(queries_with_params, desc="执行 Cypher 语句"):
                            '''
                            transaction_logic(tx)事务函数
                            tx事务 在事务中执行 如果有错误 会回滚事务
                            保证数据的绝对安全

                            query查询语句 主要查询三样 功效 和 疾病/症状
                            要把这些疾病/症状的名字作为向量存入 FAISS，供后续相似度检索。比如"感冒"、"发烧"、"咳嗽"都会存进去。
                            '''
                            tx.run(query, params or {})
                    session.execute_write(transaction_logic)
                    print(f"✅ 第 {attempt} 次执行成功")
                    return  # 执行成功，直接返回

            except Exception as e:
                # 记录错误信息
                last_error = e
                print(f"⚠️ 第 {attempt}/{max_retries} 次执行失败: {e}")

                if attempt < max_retries:
                    print(f"🔁 正在使用 LLM 修复 Cypher 查询...")
                    # 调用 LLM 修复查询（直接修改传入的列表）
                    self._fix_cypher_with_llm(queries_with_params, str(e))
                    print(f"🔄 修复后共 {len(queries_with_params)} 条查询，准备重试...")
                else:
                    print(f"❌ 第 {attempt}/{max_retries} 次执行失败，已达最大重试次数")
                    print(f"📋 错误日志: {e}")
                    print(f"📋 待执行的查询数量: {len(queries_with_params)}")
                    print(f"📋 前10条查询内容:")
                    for i, (q, p) in enumerate(queries_with_params[:10], 1):
                        print(f"   [{i}] {q[:150]}...")
                        print(f"       params: {p}")
                    if len(queries_with_params) > 10:
                        print(f"   ... 还有 {len(queries_with_params) - 10} 条查询未打印")
                    raise RuntimeError(
                        f"Neo4j 事务执行失败，已重试 {max_retries} 次仍失败。请人工检查上述日志并修复问题。"
                    ) from last_error

    def get_tcm_node_labels(self):
        """获取所有 project = 'TCM' 的节点标签"""
        # 查询 TCM 节点下所有 标签 （包括功效、疾病、症状）
        query = """
        MATCH (n)
        WHERE n.project = "TCM"
        UNWIND labels(n) AS label
        RETURN DISTINCT label
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record["label"] for record in result]

    def get_tcm_relationship_structure(self):
        """
        获取 project='TCM' 的节点参与的三元组结构：
        (起始节点标签, 关系类型, 终止节点标签)
        """
        # 找到图里所有的 三元组 （n起始节点标签, r关系类型, m终止节点标签）
        # 保留 TCM 节点 项目的 三元组结构 过滤无关数据 获取 n r m
        # head(labels(n))获取所有标签的 第一个标签 作为 from_label起始节点标签
        # type(r)获取关系类型 重命名 rel_type
        # head(labels(m)) AS to_label 获取终止节点标签
        # distinct 去重 三元组结构
        query = """
        MATCH (n)-[r]->(m)
        WHERE n.project = "TCM"
        WITH head(labels(n)) AS from_label, type(r) AS rel_type, head(labels(m)) AS to_label
        RETURN DISTINCT from_label, rel_type, to_label
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

    def export_tcm_metadata_to_json(self, output_path="tcm_metadata.json"):
        with self.driver.session() as session:

            # 1. 所有去重节点标签
            label_query = """
            MATCH (n)
            WHERE n.project = "TCM"
            UNWIND labels(n) AS label
            RETURN DISTINCT label
            """
            labels = [record["label"] for record in session.run(label_query)]

            # 2. 所有去重关系类型
            rel_query = """
            MATCH (n)-[r]-()
            WHERE n.project = "TCM"
            RETURN DISTINCT type(r) AS rel_type
            """
            rel_types = [record["rel_type"] for record in session.run(rel_query)]

            # 3. 所有去重三元组结构 distinct
            triple_query = """
            MATCH (n)-[r]->(m)
            WHERE n.project = "TCM"
            WITH head(labels(n)) AS from_label, type(r) AS rel_type, head(labels(m)) AS to_label
            RETURN DISTINCT from_label, rel_type, to_label
            """
            triples = [{
                "from": record["from_label"],
                "rel_type": record["rel_type"],
                "to": record["to_label"],
                "description": ""
            } for record in session.run(triple_query)]

            # 4. 节点属性（每个标签下的属性键）
            # unwind 把标签列表拆成一行一行
            # labels 标签 keys 属性  ORDER BY  排序
            node_props_query = """
            MATCH (n)
            WHERE n.project = "TCM"
            UNWIND labels(n) AS label
            UNWIND keys(n) AS prop
            RETURN DISTINCT label, prop
            ORDER BY label, prop
            """
            label_props = {}
            for record in session.run(node_props_query):
                label = record["label"]
                prop = record["prop"]
                if prop == "project":  # 忽略 project 字段
                    continue
                label_props.setdefault(label, []).append({
                    "name": prop,
                    "description": ""
                })

            # 5. 关系属性（每种关系下的属性键）
            rel_props_query = """
            MATCH (n)-[r]->(m)
            WHERE n.project = "TCM"
            UNWIND keys(r) AS prop
            RETURN DISTINCT type(r) AS rel_type, prop
            ORDER BY rel_type, prop
            """
            rel_type_props = {}
            for record in session.run(rel_props_query):
                rel_type = record["rel_type"]
                prop = record["prop"]
                rel_type_props.setdefault(rel_type, []).append({
                    "name": prop,
                    "description": ""
                })

            # 构建 JSON
            json_obj = {
                "labels": [
                    {
                        "name": label,
                        "description": "",
                        "properties": label_props.get(label, [])
                    } for label in labels
                ],
                "relationships": [
                    {
                        "type": rel,
                        "description": "",
                        "properties": rel_type_props.get(rel, [])
                    } for rel in rel_types
                ],
                "triples": triples
            }

            # 保存到文件
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, ensure_ascii=False, indent=2)

            return output_path
    #
    # 从数据库中查询所有疾病和症状的名称和标签
    def get_all_disease_and_symptom_names(self):
        query = """
        MATCH (n)
        WHERE n:Symptom OR n:Disease
        RETURN n.name AS name, labels(n)[0] AS label
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [(record["name"], record["label"]) for record in result]
    #
    # 从数据库中查询所有功效的名称和标签
    def get_all_effect_names(self):
        query = """
        MATCH (n)
        WHERE n:Effect
        RETURN n.name AS name, labels(n)[0] AS label
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [(record["name"], record["label"]) for record in result]

    def __del__(self):
        self.close()


neo4j_client = Neo4jClient(conf.NEO4J_URI, conf.NEO4J_USER, conf.NEO4J_PASSWORD)
