import atexit
import os

from dotenv import load_dotenv
from langchain.tools import tool

from neo4j import GraphDatabase

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD must be set in .env file")

# Initialize driver with proper configuration
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    max_connection_lifetime=3600,
    max_connection_pool_size=50,
    connection_acquisition_timeout=120
)

# Ensure a driver is closed on exit
def close_driver():
    if driver:
        driver.close()
        print("Neo4j driver closed.")

atexit.register(close_driver)


@tool
def find_similar_functions(function_signature: str, top_k: int = 5) -> str:
    print(f"Finding similar functions for: {function_signature}")
    query = """
    MATCH (f:Function {signature: $sig})
    CALL db.index.vector.queryNodes('functionCodeIndex', $top_k, f.embedding) 
    YIELD node AS neighbor, score
    WHERE f <> neighbor
    RETURN neighbor.id as id, neighbor.signature as signature, score
    ORDER BY score DESC
    """
    try:
        with driver.session() as session:
            results = session.run(query, sig=function_signature, top_k=top_k + 1)
            functions = [dict(record) for record in results]

            if not functions:
                return f"No similar functions found for '{function_signature}'. The function may not exist or have embeddings."

            return str(functions)
    except Exception as e:
        return f"Error finding similar functions: {str(e)}"


@tool
def find_callers(function_signature: str) -> str:
    print(f"Finding callers for: {function_signature}")
    query = """
    MATCH (caller:Function)-[:CALLS]->(callee:Function {signature: $sig})
    RETURN caller.id as id, caller.signature as signature
    ORDER BY caller.signature
    """
    try:
        with driver.session() as session:
            results = session.run(query, sig=function_signature)
            callers = [dict(record) for record in results]

            if not callers:
                return f"No callers found for '{function_signature}'. This function may not be called by any other function, or it may not exist in the graph."

            return str(callers)
    except Exception as e:
        return f"Error finding callers: {str(e)}"


@tool
def find_callees(function_signature: str) -> str:
    print(f"Finding callees for: {function_signature}")
    query = """
    MATCH (caller:Function {signature: $sig})-[:CALLS]->(callee:Function)
    RETURN callee.id as id, callee.signature as signature
    ORDER BY callee.signature
    """
    try:
        with driver.session() as session:
            results = session.run(query, sig=function_signature)
            callees = [dict(record) for record in results]

            if not callees:
                return f"No callees found for '{function_signature}'. This function may not call any other functions, or it may not exist in the graph."

            return str(callees)
    except Exception as e:
        return f"Error finding callees: {str(e)}"


def verify_connection() -> bool:
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            return result.single()['test'] == 1
    except Exception as e:
        print(f"Neo4j connection failed: {e}")
        return False
