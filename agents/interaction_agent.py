import logging
import re
from typing import Dict, Any, List
from retrieval.rag_pipeline import LocalMedicalRAGPipeline

logger = logging.getLogger(__name__)

class DrugInteractionAgent:
    def __init__(self, rag_pipeline: LocalMedicalRAGPipeline):
        self.pipeline = rag_pipeline

    def _extract_drugs(self, query: str) -> List[str]:
        """
        Extracts potential drug names from the query.
        For simplicity, looks for words that are not stop words.
        """
        query_lower = query.lower()
        # Clean query
        query_clean = re.sub(r'[?,.!\-()]', ' ', query_lower)
        words = query_clean.split()
        
        stop_words = {
            "what", "is", "an", "are", "the", "and", "for", "with", "does", "interact", 
            "warning", "warnings", "contraindications", "dosage", "administration", 
            "side", "effects", "adverse", "reactions", "interactions", "drug", "drugs",
            "information", "about", "effects", "side", "can", "be", "taken", 
            "together", "mixing", "mix", "safe", "safety", "take", "coadministration",
            "medicine", "medicines", "tablets", "capsules", "or", "to", "between", "check"
        }
        
        drugs = [w.strip() for w in words if len(w) > 2 and w not in stop_words]
        return list(set(drugs))

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes drug interaction queries by targeted retrieval for both drugs.
        """
        query = state["query"]
        logger.info(f"DrugInteractionAgent processing query: '{query}'")
        
        drugs = self._extract_drugs(query)
        logger.info(f"Extracted drugs for interaction check: {drugs}")
        
        if len(drugs) < 2:
            # Fallback to standard RAG pipeline if we can't extract at least two drugs
            logger.info("Fewer than 2 drugs extracted. Falling back to general pipeline...")
            res = self.pipeline.answer_query(query)
            return {
                "agent_response": res["answer"],
                "verified_sources": res.get("sources", []),
                "citations_validated": res.get("citations_validated", False),
                "final_answer": res["answer"]
            }
            
        # Target retrieval: retrieve drug interactions and warnings sections for each drug
        all_candidates = []
        for drug in drugs:
            # Perform a vector + keyword search specifically for this drug's interactions
            search_query = f"{drug} drug interactions warnings contraindications"
            candidates = self.pipeline.retriever.retrieve(search_query, limit=8)
            all_candidates.extend(candidates)
            
        # Deduplicate candidates
        seen_ids = set()
        dedup_candidates = []
        for c in all_candidates:
            if c["id"] not in seen_ids:
                seen_ids.add(c["id"])
                dedup_candidates.append(c)
                
        if not dedup_candidates:
            return {
                "agent_response": "No interaction information was found in the database for the specified drugs.",
                "verified_sources": [],
                "citations_validated": True,
                "final_answer": "No interaction information was found in the database."
            }
            
        # Rerank candidates against the original interaction query
        reranked = self.pipeline.reranker.rerank(query, dedup_candidates, top_k=4)
        
        # Filter for strict name relevance to prevent false matching
        # Make sure that the retrieved documents belong to at least one of the queried drugs
        synonyms_map = {
            "paracetamol": ["acetaminophen"],
            "acetaminophen": ["paracetamol"],
            "aspirin": ["acetylsalicylic"],
            "acetylsalicylic": ["aspirin"]
        }
        expanded_drugs = []
        for d in drugs:
            expanded_drugs.append(d)
            if d.lower() in synonyms_map:
                expanded_drugs.extend(synonyms_map[d.lower()])
        drugs_for_matching = list(set(expanded_drugs))
        
        filtered = []
        for chunk in reranked:
            brand = chunk["metadata"].get("brand_name", "").lower()
            generic = chunk["metadata"].get("generic_name", "").lower()
            if any(d in brand or d in generic for d in drugs_for_matching):
                filtered.append(chunk)
                
        if not filtered:
            return {
                "agent_response": f"No specific clinical interaction data was found for the combination of: {', '.join(drugs).title()}.",
                "verified_sources": [],
                "citations_validated": True,
                "final_answer": "No interaction data found in database."
            }
            
        # Build prompt specialized for interactions
        system_prompt = (
            "You are PharmaAssist DSLM, a professional Drug Interaction Checker.\n"
            "Analyze whether the specified drugs can be taken together based strictly on the provided context.\n"
            "Detail any known clinical interactions, mechanisms of action, warnings, or contraindications.\n"
            "Cite statements using '[Doc X]' tags matching the context index. If no interaction is mentioned in the context, explicitly state that no interaction was found in the database."
        )
        
        user_prompt = self.pipeline.generate_user_prompt(query, filtered)
        
        try:
            raw_response = self.pipeline.client.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1
            )
        except Exception as e:
            logger.error(f"Interaction LLM Error: {e}")
            return {
                "agent_response": f"Interaction check error: {str(e)}",
                "verified_sources": [],
                "citations_validated": False,
                "final_answer": f"Error: {str(e)}"
            }
            
        # Validate citations
        clean_answer, verified_sources, all_citations_valid = self.pipeline.validator.validate_and_enrich(
            raw_response, 
            filtered
        )
        
        return {
            "agent_response": clean_answer,
            "verified_sources": verified_sources,
            "citations_validated": all_citations_valid,
            "final_answer": clean_answer
        }
