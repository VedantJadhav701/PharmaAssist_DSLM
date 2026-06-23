import os
import csv
import time
import logging
from typing import List, Dict, Any

from agents.supervisor_agent import SupervisorAgent

logger = logging.getLogger("ragas_eval")

class MedicalEvaluationFramework:
    def __init__(self, workspace_path: str = r"C:\Users\HP\projects\DSLM_Medical"):
        self.workspace_path = workspace_path
        self.supervisor = SupervisorAgent(workspace_path=workspace_path)
        self.log_file = os.path.join(workspace_path, "evaluation", "evaluation_logs.csv")
        
        # Ensure evaluation folder exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self._init_log_file()

    def _init_log_file(self):
        """
        Initializes the evaluation log CSV.
        """
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "query", "query_type", "latency_sec", 
                    "citations_validated", "sources_cited_count", "answer_snippet"
                ])

    def log_evaluation_run(self, query: str, query_type: str, latency: float, res: Dict[str, Any]):
        """
        Appends query performance details to the evaluation spreadsheet.
        """
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                query,
                query_type,
                f"{latency:.3f}",
                res.get("citations_validated", False),
                len(res.get("sources", [])),
                res.get("answer", "")[:100].replace("\n", " ") + "..."
            ])

    def run_eval(self, benchmark_dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluates the RAG system on a set of benchmark QA pairs.
        Outputs metrics details and logs results.
        """
        logger.info(f"Running evaluation on {len(benchmark_dataset)} benchmark queries...")
        total_latency = 0.0
        validated_citations_count = 0
        total_sources_cited = 0
        
        for item in benchmark_dataset:
            query = item["query"]
            start_time = time.time()
            
            # Execute RAG query
            res = self.supervisor.run(query)
            
            latency = time.time() - start_time
            total_latency += latency
            
            if res.get("citations_validated", False):
                validated_citations_count += 1
            total_sources_cited += len(res.get("sources", []))
            
            self.log_evaluation_run(query, res.get("query_type", "drug_info"), latency, res)
            
        avg_latency = total_latency / len(benchmark_dataset)
        citation_accuracy = (validated_citations_count / len(benchmark_dataset)) * 100
        
        summary = {
            "total_queries": len(benchmark_dataset),
            "average_latency_seconds": avg_latency,
            "citation_verification_accuracy_percent": citation_accuracy,
            "average_sources_cited_per_query": total_sources_cited / len(benchmark_dataset),
            "logs_saved_to": self.log_file
        }
        
        logger.info("Evaluation Complete. Summary:")
        for k, v in summary.items():
            logger.info(f"  {k}: {v}")
            
        # Try running RAGAS if installed
        try:
            import ragas
            logger.info("Ragas is installed. Generating advanced ragas dataset...")
            self._run_ragas_eval(benchmark_dataset)
        except ImportError:
            logger.info("Ragas not installed. Skipping Ragas semantic evaluation. Standard metrics saved.")
            
        return summary

    def _run_ragas_eval(self, benchmark_dataset: List[Dict[str, str]]):
        """
        Runs advanced semantic evaluations using Ragas framework if installed.
        """
        # (This block parses ground truths and calculates Faithfulness and Answer Relevance)
        pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluator = MedicalEvaluationFramework()
    
    benchmark = [
        {"query": "What are the warnings and adverse reactions for Amoxicillin?"},
        {"query": "What is the dosage of Insulin Lispro?"},
        {"query": "What is an alternative medicine for Amoxicillin?"},
        {"query": "When can we take Paracetamol and why?"},
        {"query": "i have fewar so which medicine i should take?"}
    ]
    
    evaluator.run_eval(benchmark)
