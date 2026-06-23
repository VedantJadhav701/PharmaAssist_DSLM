import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MedicalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading embedding model {model_name} on {device}...")
        
        # Load the SentenceTransformer model
        # We explicitly set the device to "cpu" by default to reserve GPU for the local LLM.
        self.model = SentenceTransformer(model_name, device=device)
        logger.info("Embedding model loaded successfully.")

    def embed_query(self, query: str) -> List[float]:
        """
        Generates embedding for a user query.
        BGE-small-en-v1.5 has an instruction prefix recommendation for queries:
        "Represent this sentence for searching relevant passages: "
        """
        # Prefix the query as recommended by BGE authors for retrieval tasks
        prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
        embedding = self.model.encode(prefixed_query, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a batch of documents.
        Do not add query prefixes for documents.
        """
        if not documents:
            return []
        embeddings = self.model.encode(documents, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embedder = MedicalEmbedder()
    test_query = "What is the dosage of Metformin?"
    emb = embedder.embed_query(test_query)
    print(f"Query embedding size: {len(emb)}")
    
    test_docs = ["Metformin is indicated for type 2 diabetes.", "Dosage is 500mg daily."]
    embs = embedder.embed_documents(test_docs)
    print(f"Docs embeddings size: {len(embs)} x {len(embs[0])}")
