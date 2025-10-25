# import_to_neo4j.py

import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm

# --- Configuration ---
load_dotenv()

# 1. Neo4j connection configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    raise ValueError("Please set NEO4J_PASSWORD in your .env file")


class Neo4jCodeGraphImporter:
    """
    Import data from a JSON file and build a code-structure graph in Neo4j.

    The importer expects the JSON file to be a list of chunk objects. Each chunk
    should contain at least an 'id', a 'content' field and a 'metadata' dict that
    may include fields like 'file_path', 'name', 'signature', 'called_by', and 'imports'.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.all_json_data = []

    def close(self):
        self.driver.close()

    def _load_data(self, json_file_path: str):
        """Load JSON data from disk.

        Raises FileNotFoundError or TypeError for invalid contents.
        """
        print(f"Reading data from: {json_file_path}")
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")

        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.all_json_data = json.load(f)

        if not isinstance(self.all_json_data, list):
            raise TypeError("The JSON file must contain a list of chunk objects.")

    def _clear_database(self):
        """Delete all nodes and relationships in the database."""
        print("Clearing existing database data...")
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))

    def _create_nodes_pass(self):
        """Pass 1: Create all nodes (Function, Module, Class) so they exist before relationships."""
        print("Pass 1: Creating nodes...")
        with self.driver.session() as session:
            for data in tqdm(self.all_json_data, desc="Creating Nodes"):
                session.execute_write(self._create_nodes_transaction, data)

    @staticmethod
    def _create_nodes_transaction(tx, data: dict):
        """Transaction to create nodes for a single JSON chunk."""
        node_id = data['id']
        metadata = data.get('metadata', {})
        file_path = metadata.get('file_path')

        # Normalize file_path to a consistent OS path style
        if file_path:
            file_path = os.path.normpath(file_path)

        # 1. MERGE Function node (including signature)
        tx.run(
            """
            MERGE (f:Function {id: $id})
            ON CREATE SET 
                f.name = $name,
                f.content = $content,
                f.signature = $signature
            """,
            id=node_id,
            name=metadata.get('name'),
            content=data.get('content'),
            signature=metadata.get('signature')
        )

        # 2. MERGE Module node
        if file_path:
            tx.run("MERGE (:Module {path: $path})", path=file_path)

        # 3. MERGE Class node (if present)
        if file_path:
            # Try to get the relative path of the node id to the file_path and split using OS sep
            try:
                rel = os.path.relpath(node_id, start=file_path)
            except Exception:
                # Fallback: remove file_path prefix if present
                rel = str(node_id).replace(file_path, '')
            id_parts = str(rel).strip(os.sep).split(os.sep)
            if len(id_parts) > 1:
                class_name = id_parts[0]
                # Create a unique class id to avoid collisions across files
                class_id = os.path.join(file_path, class_name)
                tx.run(
                    "MERGE (c:Class {id: $id}) ON CREATE SET c.name = $name, c.file_path = $file_path",
                    id=class_id, name=class_name, file_path=file_path
                )

    def _create_edges_pass(self):
        """Pass 2: Create all relationships (CALLS, CONTAINS, INCLUDES)."""
        print("Pass 2: Creating relationships...")
        with self.driver.session() as session:
            for data in tqdm(self.all_json_data, desc="Creating Edges"):
                session.execute_write(self._create_edges_transaction, data)

    @staticmethod
    def _create_edges_transaction(tx, data: dict):
        """Transaction to create relationships for a single JSON chunk."""
        callee_id = data['id']
        metadata = data.get('metadata', {})
        file_path = metadata.get('file_path')

        if file_path:
            file_path = os.path.normpath(file_path)

        # 1. CONTAINS relationships
        if file_path:
            # Module -> Function
            tx.run(
                "MATCH (m:Module {path: $path}), (f:Function {id: $id}) MERGE (m)-[:CONTAINS]->(f)",
                path=file_path, id=callee_id
            )

            # Class -> Function and Module -> Class
            try:
                rel = os.path.relpath(callee_id, start=file_path)
            except Exception:
                rel = str(callee_id).replace(file_path, '')
            id_parts = str(rel).strip(os.sep).split(os.sep)
            if len(id_parts) > 1:
                class_name = id_parts[0]
                class_id = os.path.join(file_path, class_name)
                tx.run(
                    "MATCH (c:Class {id: $id}), (f:Function {id: $func_id}) MERGE (c)-[:CONTAINS]->(f)",
                    id=class_id, func_id=callee_id
                )
                tx.run(
                    "MATCH (m:Module {path: $path}), (c:Class {id: $id}) MERGE (m)-[:CONTAINS]->(c)",
                    path=file_path, id=class_id
                )

        # 2. CALLS relationships (process `called_by`)
        if 'called_by' in metadata and metadata.get('called_by'):
            for caller_id in metadata['called_by']:
                tx.run(
                    "MATCH (caller:Function {id: $caller_id}), (callee:Function {id: $callee_id}) MERGE (caller)-[:CALLS]->(callee)",
                    caller_id=caller_id, callee_id=callee_id
                )

        # 3. INCLUDES relationships (module imports)
        if file_path and 'imports' in metadata and metadata.get('imports'):
            for target_path in metadata['imports']:
                target_path_norm = os.path.normpath(target_path)
                tx.run("MERGE (:Module {path: $path})", path=target_path_norm)
                tx.run(
                    """
                    MATCH (source:Module {path: $source}), (target:Module {path: $target})
                    MERGE (source)-[:INCLUDES]->(target)
                    """,
                    source=file_path, target=target_path_norm
                )

    def build_graph(self, json_file_path: str):
        """Run the full import: load data, clear DB, create nodes, and create relationships."""
        self._load_data(json_file_path)
        self._clear_database()
        self._create_nodes_pass()
        self._create_edges_pass()
        print("\n--- Code structure graph import to Neo4j completed! ---")


# --- Usage ---
if __name__ == "__main__":
    JSON_FILE = './chunks_output.json'
    importer = None
    try:
        importer = Neo4jCodeGraphImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        importer.build_graph(JSON_FILE)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if importer:
            importer.close()