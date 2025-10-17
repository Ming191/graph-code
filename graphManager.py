# graph_manager.py

import os
import json
import numpy as np
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config


class CodeGraphManager:
    """Quản lý tất cả các tương tác với cơ sở dữ liệu đồ thị Neo4j."""

    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("✅ Kết nối đến Neo4j thành công!")
        except Exception as e:
            print(f"❌ Lỗi kết nối đến Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()
            print("🔌 Đã đóng kết nối Neo4j.")

    def run_query(self, query, parameters=None, **kwargs):
        """Execute a Cypher query with parameters.

        Args:
            query: The Cypher query string
            parameters: Dictionary of parameters (optional)
            **kwargs: Individual parameters as keyword arguments

        Returns:
            List of record dictionaries
        """
        # Merge parameters from both dict and kwargs
        if parameters is None:
            parameters = {}
        parameters.update(kwargs)

        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def clear_database(self):
        print("🗑️ Đang xóa toàn bộ dữ liệu trong cơ sở dữ liệu Neo4j...")
        self.run_query("MATCH (n) DETACH DELETE n")
        print("✅ Xóa dữ liệu thành công.")

    def build_ds_code_graph_from_single_file(self, json_file_path):
        """Đọc một tệp JSON chứa danh sách các chunk và xây dựng đồ thị."""
        if not self.driver: return

        print("\n--- Bắt đầu xây dựng DS-Code Graph từ một tệp duy nhất ---")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                all_json_data = json.load(f)
            if not isinstance(all_json_data, list):
                raise TypeError("Tệp JSON phải chứa một danh sách (list) các đối tượng chunk.")
        except FileNotFoundError:
            print(f"❌ Lỗi: Không tìm thấy tệp '{json_file_path}'.")
            return
        except (json.JSONDecodeError, TypeError) as e:
            print(f"❌ Lỗi khi đọc tệp JSON: {e}")
            return

        all_relationships = []
        for data in all_json_data:
            if 'metadata' not in data or 'id' not in data: continue

            self.run_query("""
                MERGE (c:CodeChunk {id: $id})
                SET c.content = $content, c.name = $metadata.name,
                    c.type = $metadata.type, c.file_path = $metadata.file_path
            """, id=data['id'], content=data['content'], metadata=data['metadata'])

            if 'calls' in data['metadata'] and data['metadata']['calls']:
                for callee_id in data['metadata']['calls']:
                    all_relationships.append({'caller': data['id'], 'callee': callee_id})

            if 'called_by' in data['metadata'] and data['metadata']['called_by']:
                for caller_id in data['metadata']['called_by']:
                    all_relationships.append({'caller': caller_id, 'callee': data['id']})

        unique_relationships = [dict(t) for t in {tuple(d.items()) for d in all_relationships}]
        for rel in unique_relationships:
            self.run_query("""
                MATCH (caller:CodeChunk {id: $caller_id})
                MATCH (callee:CodeChunk {id: $callee_id})
                MERGE (caller)-[:CALLS]->(callee)
            """, caller_id=rel['caller'], callee_id=rel['callee'])
        print(f"✅ Hoàn thành DS-Code Graph. Đã thêm {len(unique_relationships)} cạnh :CALLS.")

    def add_similarity_edges(self, threshold=0.85):
        """Tạo embeddings sử dụng TF-IDF và thêm các cạnh :SIMILARITY."""
        if not self.driver: return
        print("\n--- Bắt đầu thêm các cạnh :SIMILARITY (sử dụng TF-IDF) ---")

        nodes_data = self.run_query("MATCH (c:CodeChunk) RETURN c.id as id, c.content as content")
        if not nodes_data: return

        node_ids = [record['id'] for record in nodes_data]
        node_contents = [record['content'] if record['content'] else "" for record in nodes_data]

        # Use TF-IDF for creating embeddings (memory efficient)
        print("📊 Đang tạo TF-IDF vectors...")
        vectorizer = TfidfVectorizer(
            max_features=500,  # Limit features for memory efficiency
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=1,
            stop_words=None  # Keep all words for code similarity
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(node_contents)
            embeddings = tfidf_matrix.toarray()
        except Exception as e:
            print(f"❌ Lỗi khi tạo TF-IDF vectors: {e}")
            return

        # Store embeddings in database
        print("💾 Đang lưu embeddings vào Neo4j...")
        for i, node_id in enumerate(node_ids):
            self.run_query("MATCH (c:CodeChunk {id: $id}) SET c.embedding = $embedding",
                           id=node_id, embedding=embeddings[i].tolist())

        # Calculate similarities in batches to save memory
        print("🔗 Đang tính toán độ tương đồng...")
        similarity_count = 0
        batch_size = 50  # Process in batches

        for i in range(0, len(node_ids), batch_size):
            end_i = min(i + batch_size, len(node_ids))
            batch_embeddings = embeddings[i:end_i]

            # Calculate similarity only with subsequent nodes
            for j in range(end_i, len(node_ids)):
                similarities = cosine_similarity(batch_embeddings, embeddings[j:j + 1])

                for local_idx, similarity_score in enumerate(similarities.flatten()):
                    if similarity_score > threshold:
                        actual_i = i + local_idx
                        self.run_query("""
                            MATCH (c1:CodeChunk {id: $id1}), (c2:CodeChunk {id: $id2})
                            MERGE (c1)-[:SIMILARITY]-(c2)
                        """, id1=node_ids[actual_i], id2=node_ids[j])
                        similarity_count += 1

        print(f"✅ Hoàn thành. Đã thêm {similarity_count} cạnh :SIMILARITY.")

    def find_initial_support_codes(self, user_requirement: str, top_k=5):
        """Thực hiện Bigraph Mapping: Tìm code hỗ trợ ban đầu."""
        if not self.driver: return ""

        print("\n--- Bắt đầu tìm kiếm code hỗ trợ ban đầu (Bigraph Mapping) ---")

        results = self.run_query(
            "MATCH (c:CodeChunk) WHERE c.embedding IS NOT NULL RETURN c.id as id, c.content as content, c.embedding as embedding")
        if not results: return ""

        db_ids = [record['id'] for record in results]
        db_contents = [record['content'] if record['content'] else "" for record in results]
        db_embeddings = np.array([record['embedding'] for record in results])

        # Create TF-IDF vector for user requirement using same vocabulary
        print("🔍 Đang tính toán độ tương đồng với yêu cầu...")
        vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=1,
            stop_words=None
        )

        # Fit on both database contents and user requirement
        all_texts = db_contents + [user_requirement]
        vectorizer.fit(all_texts)

        # Transform user requirement
        req_vector = vectorizer.transform([user_requirement]).toarray()

        # Transform database contents
        db_vectors = vectorizer.transform(db_contents).toarray()

        # Calculate similarities
        similarities = cosine_similarity(req_vector, db_vectors)[0]

        target_node_id = db_ids[np.argmax(similarities)]
        print(f"Nút phù hợp nhất với yêu cầu: {target_node_id} (similarity: {np.max(similarities):.4f})")

        query = """
            MATCH (target:CodeChunk {id: $target_id})
            OPTIONAL MATCH (target)-[r:CALLS|:SIMILARITY]-(neighbor)
            WITH target, COLLECT(neighbor) AS neighbors
            UNWIND ([target] + neighbors) AS node
            RETURN DISTINCT node.content AS content, node.id as id
            LIMIT $limit
        """
        support_nodes = self.run_query(query, target_id=target_node_id, limit=top_k)

        context = "/* --- Code hỗ trợ được tìm thấy trong kho code --- */\n\n"
        for node in support_nodes:
            context += f"// From: {node['id']}\n{node['content']}\n\n"

        print(f"✅ Đã tìm thấy {len(support_nodes)} đoạn code hỗ trợ.")
        return context