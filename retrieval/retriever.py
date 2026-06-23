import logging
from typing import List, Dict, Any, Optional
from embeddings.embedder import MedicalEmbedder
from vectorstore.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, 
                 workspace_path: str = r"C:\Users\HP\projects\DSLM_Medical",
                 collection_name: str = "pharmaassist_medical"):
        self.embedder = MedicalEmbedder(model_name="BAAI/bge-small-en-v1.5", device="cpu")
        self.qdrant = QdrantManager(collection_name=collection_name, path=f"{workspace_path}/qdrant_db")

    def retrieve(self, query: str, limit: int = 15, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves candidate documents using hybrid search:
        Combines semantic vector search and exact keyword search.
        Deduplicates results and keeps top scoring candidates.
        """
        logger.info(f"Retrieving candidate documents for query: '{query}'...")
        
        # 1. Semantic Vector Search
        query_vector = self.embedder.embed_query(query)
        semantic_results = self.qdrant.semantic_search(
            query_vector=query_vector, 
            limit=limit, 
            filter_dict=filter_dict
        )
        
        # 2. Keyword Search (for exact name matches)
        # Extract terms from query to match brand or generic name
        keyword_results = []
        words = [w.strip(",.!?\"'") for w in query.split() if len(w) > 2]
        
        # Expand synonyms for keyword search
        synonyms_map = {
            "paracetamol": ["acetaminophen"],
            "acetaminophen": ["paracetamol"],
            "aspirin": ["acetylsalicylic"],
            "acetylsalicylic": ["aspirin"]
        }
        
        expanded_words = []
        for w in words:
            w_lower = w.lower()
            expanded_words.append(w)
            if w_lower in synonyms_map:
                expanded_words.extend(synonyms_map[w_lower])
        
        for word in list(set(expanded_words)):
            # Skip common question words or short words
            if word.lower() in ["what", "how", "with", "does", "interact", "warning", "contraindication", "dosage", "side", "effect", "use", "for"]:
                continue
            word_matches = self.qdrant.keyword_search(word, limit=3)
            keyword_results.extend(word_matches)
            
        # 3. Combine & Deduplicate
        seen_ids = set()
        combined = []
        
        # Give higher initial preference to exact keyword search results if they exist
        # by putting them first, but maintaining similarity scores
        for res in keyword_results:
            point_id = res["id"]
            if point_id not in seen_ids:
                seen_ids.add(point_id)
                # Assign a booster score for exact matching
                res["score"] = 0.95
                combined.append(res)
                
        for res in semantic_results:
            point_id = res["id"]
            if point_id not in seen_ids:
                seen_ids.add(point_id)
                combined.append(res)
                
        # Limit combined list to the requested size
        logger.info(f"Retrieved {len(combined)} deduplicated hybrid search candidates.")
        return combined[:limit]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retriever = HybridRetriever()
    results = retriever.retrieve("Metformin warnings")
    print(f"Retrieved {len(results)} items.")
    for idx, r in enumerate(results[:3]):
        print(f"[{idx}] {r['metadata']['brand_name']} - {r['metadata']['section']} (Score: {r['score']:.4f})")
