import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class MedicalReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading cross-encoder reranker {model_name} on {device}...")
        
        # Load the CrossEncoder model
        # Explicitly configure to run on CPU to avoid CUDA OOM.
        self.model = CrossEncoder(model_name, device=device)
        logger.info("Reranker model loaded successfully.")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate documents using the CrossEncoder model.
        Returns the top_k scoring documents.
        """
        if not candidates:
            return []
            
        logger.info(f"Reranking {len(candidates)} candidates down to {top_k}...")
        
        # Prepare inputs for cross encoder: pairs of (query, document_text)
        pairs = [[query, cand["text"]] for cand in candidates]
        
        # Compute relevance scores
        scores = self.model.predict(pairs)
        
        # Assign scores to candidates
        for cand, score in zip(candidates, scores):
            # The output of CrossEncoder can be converted to float
            cand["rerank_score"] = float(score)
            
        # Sort candidates by rerank score descending
        ranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        # Select top k
        top_candidates = ranked_candidates[:top_k]
        logger.info(f"Reranking complete. Top score: {top_candidates[0]['rerank_score']:.4f} if items present.")
        return top_candidates

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reranker = MedicalReranker()
    test_query = "What is the dose of Metformin?"
    test_cands = [
        {"text": "Metformin side effects include nausea and diarrhea.", "id": "1"},
        {"text": "For Type 2 Diabetes, Metformin dosage starts at 500mg daily.", "id": "2"},
        {"text": "Tylenol is an analgesic.", "id": "3"}
    ]
    ranked = reranker.rerank(test_query, test_cands, top_k=2)
    for idx, r in enumerate(ranked):
        print(f"[{idx}] (Score: {r['rerank_score']:.4f}) - {r['text']}")
