import logging
from typing import Dict, Any, List
from retrieval.retriever import HybridRetriever
from retrieval.reranker import MedicalReranker
from retrieval.citation_validator import CitationValidator
from llm.qwen_client import QwenClient

logger = logging.getLogger(__name__)

class LocalMedicalRAGPipeline:
    def __init__(self, 
                 workspace_path: str = r"C:\Users\HP\projects\DSLM_Medical",
                 collection_name: str = "pharmaassist_medical",
                 model_name: str = None):
        self.retriever = HybridRetriever(workspace_path, collection_name)
        self.reranker = MedicalReranker(device="cpu")
        self.client = QwenClient(model_name=model_name)
        self.validator = CitationValidator()

    def generate_system_prompt(self) -> str:
        return (
            "You are PharmaAssist DSLM, a professional healthcare domain assistant designed for pharmacists and medical staff.\n"
            "Your goal is to answer the user's pharmaceutical question accurately, objectively, and strictly based on the provided retrieved medical context.\n\n"
            "Strict Instructions:\n"
            "1. Base your answer ONLY on the provided context. If the context does not contain the answer, explicitly state that the information was not found in the local database.\n"
            "2. Never make up facts, side effects, or drug interactions. Factual medical accuracy is critical.\n"
            "3. For every claim or clinical statement you make, append the corresponding citation index from the context, e.g. '[Doc 1]', '[Doc 2]', etc. at the end of the sentence.\n"
            "4. Format the final output in clear, professional medical terms with bullet points where appropriate.\n"
            "5. Do NOT include a separate 'Sources' or 'References' section at the end of your response, just use the inline '[Doc X]' citations."
        )

    def generate_user_prompt(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        context_str = ""
        for idx, chunk in enumerate(chunks):
            # We number them 1-based for the LLM to reference as [Doc 1], [Doc 2] etc.
            meta = chunk["metadata"]
            context_str += (
                f"--- [Doc {idx + 1}] ---\n"
                f"Drug Brand Name: {meta.get('brand_name')}\n"
                f"Generic Name: {meta.get('generic_name')}\n"
                f"Section: {meta.get('section', '').replace('_', ' ').title()}\n"
                f"Content:\n{chunk['raw_content']}\n\n"
            )
            
        return (
            f"User Question: {question}\n\n"
            f"Retrieved Medical Context:\n"
            f"{context_str}\n"
            f"Answer the question using the context above. Remember to cite using '[Doc X]' tags at the end of each sentence."
        )

    def verify_relevance(self, query: str, chunks: List[Dict[str, Any]], rerank_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Determines if retrieved chunks are actually relevant to the query drug.
        Filters by cross-encoder score threshold AND exact drug name matching.
        """
        query_lower = query.lower()
        stop_words = {
            "what", "are", "the", "and", "for", "with", "does", "interact", 
            "warning", "warnings", "indications", "dosage", "administration", 
            "side", "effects", "adverse", "reactions", "contraindications", 
            "about", "is", "of", "to", "in", "a", "an", "or", "by", "this", 
            "that", "from", "on", "at", "but", "not", "have", "has", "had", 
            "be", "been", "was", "were", "should", "could", "would", "can", 
            "will", "may", "might", "must", "vaccine", "medicine", "tablets", 
            "capsules", "injection", "oral", "treatment", "prevent", "prevention",
            "use", "used", "indicated", "contraindicated", "safe", "safety", "take",
            "when", "why", "who", "how", "which", "should", "could", "would", "drug",
            "drugs", "medicine", "medicines", "fever", "fewar", "pain", "headache",
            "cough", "cold", "flu", "migraine", "diabetes", "hypertension", "infection",
            "constipation", "diarrhea", "nausea", "vomiting", "allergy", "allergies",
            "asthma", "depression", "anxiety", "insomnia", "heartburn", "acid", "reflux",
            "ache", "aches", "sore", "throat", "congestion", "runny", "nose", "sneezing",
            "coughing", "feverish", "chills", "illness", "sickness", "disease",
            "condition", "symptom", "symptoms"
        }
        
        # Extract potential drug terms from query
        query_words = [w.strip("?,.!-()\"'") for w in query_lower.split()]
        query_words = [w for w in query_words if len(w) > 2 and w not in stop_words]
        
        # Expand synonyms to prevent name alignment mismatches (e.g. paracetamol <-> acetaminophen)
        synonyms_map = {
            "paracetamol": ["acetaminophen"],
            "acetaminophen": ["paracetamol"],
            "aspirin": ["acetylsalicylic"],
            "acetylsalicylic": ["aspirin"]
        }
        expanded_words = []
        for w in query_words:
            expanded_words.append(w)
            if w in synonyms_map:
                expanded_words.extend(synonyms_map[w])
        query_words = list(set(expanded_words))
        
        filtered = []
        for chunk in chunks:
            # 1. Check cross-encoder score (filter completely low scores)
            score = chunk.get("rerank_score", 0.0)
            if score < rerank_threshold:
                chunk_id = chunk.get("metadata", {}).get("chunk_id", chunk.get("id"))
                logger.info(f"Chunk {chunk_id} filtered due to low rerank score ({score:.4f} < {rerank_threshold})")
                continue
                
            # 2. Check drug name alignment
            brand = chunk["metadata"].get("brand_name", "").lower()
            generic = chunk["metadata"].get("generic_name", "").lower()
            
            # If query contains any of the query words matching brand or generic names
            name_matched = False
            if not query_words:
                name_matched = True  # Fallback if query contains only stop words
            else:
                for word in query_words:
                    if word in brand or word in generic:
                        name_matched = True
                        break
                    if brand and brand in query_lower:
                        name_matched = True
                        break
                    if generic and generic in query_lower:
                        name_matched = True
                        break
                        
            if name_matched:
                filtered.append(chunk)
            else:
                chunk_id = chunk.get("metadata", {}).get("chunk_id", chunk.get("id"))
                logger.info(f"Chunk {chunk_id} filtered due to drug name mismatch (Brand: '{brand}', Generic: '{generic}')")
                
        return filtered

    def answer_query(self, query: str, limit_candidates: int = 15, limit_final: int = 4) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline:
        Retrieve -> Rerank -> Filter -> LLM Generation -> Citation Verification.
        """
        # 1. Retrieve candidates using Hybrid Search
        candidates = self.retriever.retrieve(query, limit=limit_candidates)
        if not candidates:
            return {
                "answer": "No relevant drug data was found in the local knowledge base.",
                "sources": [],
                "success": False
            }
            
        # 2. Rerank candidates using Cross-Encoder on CPU
        reranked_chunks = self.reranker.rerank(query, candidates, top_k=limit_final)
        
        # 3. Filter candidates for strict drug-relevance to prevent hallucinations
        filtered_chunks = self.verify_relevance(query, reranked_chunks, rerank_threshold=-0.5)
        if not filtered_chunks:
            return {
                "answer": "The requested drug was not found in the local knowledge base, so no information can be provided.",
                "sources": [],
                "success": False
            }
            
        # 4. Build prompts
        system_prompt = self.generate_system_prompt()
        user_prompt = self.generate_user_prompt(query, filtered_chunks)
        
        # 4. Generate response using Qwen (local Ollama)
        try:
            raw_response = self.client.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1
            )
        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            return {
                "answer": f"Inference error: {str(e)}",
                "sources": [],
                "success": False
            }
            
        # 5. Programmatically validate and enrich citations
        clean_answer, verified_sources, all_citations_valid = self.validator.validate_and_enrich(
            raw_response, 
            filtered_chunks
        )
        
        return {
            "answer": clean_answer,
            "sources": verified_sources,
            "citations_validated": all_citations_valid,
            "success": True
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = LocalMedicalRAGPipeline()
    # Test query
    print("Testing query: 'What are the warnings and adverse reactions for Amoxicillin?'")
    try:
        res = pipeline.answer_query("What are the warnings and adverse reactions for Amoxicillin?")
        print("\n=== ANSWER ===")
        print(res["answer"])
        print("\n=== VERIFIED SOURCES ===")
        for s in res["sources"]:
            print(f"- {s['brand_name']} ({s['generic_name']}) - Section: {s['section']}")
    except Exception as e:
        print("Pipeline run failed:", e)
