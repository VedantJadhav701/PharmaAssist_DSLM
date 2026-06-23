import logging
import re
from typing import Dict, Any, List
from ingestion.who_loader import WHOLoader
from retrieval.rag_pipeline import LocalMedicalRAGPipeline

logger = logging.getLogger(__name__)

class AlternativeMedicineAgent:
    def __init__(self, rag_pipeline: LocalMedicalRAGPipeline, who_loader: WHOLoader):
        self.pipeline = rag_pipeline
        self.who_loader = who_loader

    def _extract_drug(self, query: str) -> str:
        """
        Extracts the queried drug name from the query.
        Prioritizes matching generic names in the WHO EML.
        """
        query_lower = query.lower()
        # Clean query
        query_clean = re.sub(r'[?,.!\-()]', ' ', query_lower)
        words = query_clean.split()
        
        # 1. Try matching words directly against WHO EML keys
        for word in words:
            if len(word) > 2 and word in self.who_loader.eml_db:
                return word
                
        # 2. Fallback stop-word based extraction
        stop_words = {
            "what", "is", "an", "are", "the", "and", "for", "with", "does", "interact", 
            "warning", "warnings", "contraindications", "dosage", "administration", 
            "side", "effects", "adverse", "reactions", "interactions", "drug", 
            "information", "about", "effects", "side", "can", "be", "taken", 
            "together", "mixing", "mix", "safe", "safety", "take", "alternative", 
            "alternatives", "instead", "of", "substitute", "substitutes", "replace",
            "replacement", "other", "options", "option", "choice", "medicine", "medicines"
        }
        
        drugs = [w.strip() for w in words if len(w) > 2 and w not in stop_words]
        return drugs[0] if drugs else ""

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes alternative drug queries using WHO EML therapeutic sections.
        """
        query = state["query"]
        logger.info(f"AlternativeMedicineAgent processing query: '{query}'")
        
        drug = self._extract_drug(query)
        logger.info(f"Extracted drug for alternative lookup: '{drug}'")
        
        if not drug:
            return {
                "agent_response": "Could not identify which drug you want alternatives for. Please ask like 'What is an alternative for Amoxicillin?'.",
                "verified_sources": [],
                "citations_validated": True,
                "final_answer": "No drug name identified."
            }
            
        # Get EML alternatives from WHOLoader
        alternatives = self.who_loader.get_alternatives(drug)
        meta = self.who_loader.get_metadata(drug)
        
        if not alternatives:
            # Fallback to general RAG if WHO EML doesn't contain the drug
            logger.info("Drug not found in WHO EML. Falling back to general RAG search...")
            res = self.pipeline.answer_query(query)
            return {
                "agent_response": res["answer"],
                "verified_sources": res.get("sources", []),
                "citations_validated": res.get("citations_validated", False),
                "final_answer": res["answer"]
            }
            
        # Formulate a structured answer detailing WHO category and related drugs
        drug_title = drug.title()
        eml_section = meta.get("eml_section", "Unknown Section")
        
        response_lines = [
            f"According to the WHO Essential Medicines List (EML), **{drug_title}** is classified under:",
            f"* **Therapeutic Category:** {eml_section}",
            f"\nHere are essential therapeutic alternatives listed in the same WHO category:"
        ]
        
        # Present top 8 alternatives
        for idx, alt in enumerate(alternatives[:8]):
            alt_name = alt["name"].title()
            atc = ", ".join(alt["atc_codes"])
            inds = ", ".join(alt["indications"])
            response_lines.append(
                f"{idx + 1}. **{alt_name}**\n"
                f"   - *ATC Codes:* {atc}\n"
                f"   - *WHO Indications:* {inds}"
            )
            
        # Try to retrieve text from local vector store for the first few alternatives to add RAG citations
        logger.info("Retrieving description chunks for the first alternative drug...")
        first_alt = alternatives[0]["name"]
        rag_query = f"{first_alt} indications dosage warnings"
        candidates = self.pipeline.retriever.retrieve(rag_query, limit=5)
        filtered = self.pipeline.verify_relevance(rag_query, candidates, rerank_threshold=-0.5)
        
        verified_sources = []
        if filtered:
            # Rerank and keep top 2
            reranked = self.pipeline.reranker.rerank(rag_query, filtered, top_k=2)
            # Add RAG summary
            response_lines.append(f"\n**Clinical Profile for {first_alt.title()} (from local databases):**")
            for i, chunk in enumerate(reranked):
                snippet = chunk["raw_content"][:300] + "..."
                brand = chunk["metadata"].get("brand_name", "N/A").title()
                response_lines.append(f"- {snippet} [Doc {i + 1}]")
                
            # Programmatically populate verified sources for citation validator
            for i, chunk in enumerate(reranked):
                meta = chunk["metadata"]
                verified_sources.append({
                    "chunk_id": chunk.get("metadata", {}).get("chunk_id", chunk.get("id")),
                    "brand_name": meta.get("brand_name", "Unknown"),
                    "generic_name": meta.get("generic_name", "Unknown"),
                    "section": meta.get("section", "General").replace("_", " ").title(),
                    "is_essential": meta.get("is_essential", False),
                    "eml_section": meta.get("eml_section", "N/A"),
                    "atc_codes": meta.get("atc_codes", "N/A"),
                    "text_snippet": chunk.get("raw_content", "")[:200] + "..."
                })
                
        final_text = "\n".join(response_lines)
        return {
            "agent_response": final_text,
            "verified_sources": verified_sources,
            "citations_validated": True,
            "final_answer": final_text
        }
