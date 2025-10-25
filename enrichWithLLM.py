import os
import torch
from transformers import RobertaTokenizer, RobertaModel
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

load_dotenv()

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    raise ValueError("Please set NEO4J_PASSWORD in your .env file")

print("Loading CodeBERT model (may take a few minutes on first run)...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.to(device)  # Move model to GPU if available
    print("Loaded CodeBERT model successfully.")
except Exception as e:
    print("Error during model loading:", e)
    exit()



def get_codebert_embedding(code_text):
    """Get an embedding vector for a code snippet using CodeBERT."""
    try:
        inputs = tokenizer(code_text, return_tensors="pt", truncation=True, max_length=512)
        # Move each tensor in the tokenizer output to the selected device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding.cpu().numpy().tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

class GraphEnricherWithCodeBERT:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _run_query(self, query, **params):
        with self.driver.session() as session:
            return session.execute_write(lambda tx: list(tx.run(query, **params)))

    def generate_code_embeddings(self):
        functions_to_embed = self._run_query(
            "MATCH (f:Function) WHERE f.content IS NOT NULL AND f.embedding IS NULL RETURN f.id as id, f.content as content")

        if not functions_to_embed:
            print("No functions to embed. Skipping...")
            return

        for record in tqdm(functions_to_embed, desc="Generating Code Embeddings"):
            embedding = get_codebert_embedding(record["content"])
            if embedding:
                self._run_query("MATCH (f:Function {id: $id}) SET f.embedding = $embedding", id=record["id"],
                                embedding=embedding)
        print("Embedding generation completed.")


if __name__ == "__main__":
    enricher = None
    try:
        enricher = GraphEnricherWithCodeBERT(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        enricher.generate_code_embeddings()
    except Exception as e:
        print("Error during enrichment:", e)
    finally:
        if enricher:
            enricher.close()