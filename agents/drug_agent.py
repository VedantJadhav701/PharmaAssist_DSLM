import logging
from typing import Dict, Any
from retrieval.rag_pipeline import LocalMedicalRAGPipeline

logger = logging.getLogger(__name__)

class DrugInfoAgent:
    def __init__(self, rag_pipeline: LocalMedicalRAGPipeline):
        self.pipeline = rag_pipeline

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes general drug info queries.
        """
        query = state["query"]
        logger.info(f"DrugInfoAgent processing query: '{query}'")
        
        # Run the standard RAG pipeline
        res = self.pipeline.answer_query(query)
        
        return {
            "agent_response": res["answer"],
            "verified_sources": res.get("sources", []),
            "citations_validated": res.get("citations_validated", False),
            "final_answer": res["answer"]
        }
