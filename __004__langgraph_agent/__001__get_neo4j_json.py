from common.neo4j_client import neo4j_client

# 从Neo4j图谱中提取TCM元数据并导出为JSON文件。
neo4j_client.export_tcm_metadata_to_json()
