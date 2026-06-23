import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START

from retrieval.rag_pipeline import LocalMedicalRAGPipeline
from ingestion.who_loader import WHOLoader
from agents.drug_agent import DrugInfoAgent
from agents.interaction_agent import DrugInteractionAgent
from agents.alternative_agent import AlternativeMedicineAgent

logger = logging.getLogger(__name__)

# Define LangGraph state schema
class AgentState(TypedDict):
    query: str
    query_type: str
    agent_response: str
    verified_sources: List[Dict[str, Any]]
    citations_validated: bool
    final_answer: str

class SupervisorAgent:
    def __init__(self, workspace_path: str = r"C:\Users\HP\projects\DSLM_Medical"):
        # 1. Initialize pipelines and loaders
        self.pipeline = LocalMedicalRAGPipeline(workspace_path=workspace_path)
        self.who_loader = WHOLoader(file_path=f"{workspace_path}/eml_export.xlsx")
        
        # 2. Initialize worker agents
        self.drug_agent = DrugInfoAgent(self.pipeline)
        self.interaction_agent = DrugInteractionAgent(self.pipeline)
        self.alternative_agent = AlternativeMedicineAgent(self.pipeline, self.who_loader)
        
        # 3. Assemble the LangGraph workflow
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        # Create StateGraph
        builder = StateGraph(AgentState)
        
        # Add Nodes
        builder.add_node("classify", self._classify_query_node)
        builder.add_node("drug_info", self._drug_info_node)
        builder.add_node("interaction", self._interaction_node)
        builder.add_node("alternative", self._alternative_node)
        builder.add_node("general", self._general_node)
        builder.add_node("validate", self._validate_node)
        
        # Set Entrypoint
        builder.add_edge(START, "classify")
        
        # Conditional routing after classification
        builder.add_conditional_edges(
            "classify",
            self._route_query,
            {
                "drug_info": "drug_info",
                "interaction": "interaction",
                "alternative": "alternative",
                "general": "general"
            }
        )
        
        # Edges leading to validation node
        builder.add_edge("drug_info", "validate")
        builder.add_edge("interaction", "validate")
        builder.add_edge("alternative", "validate")
        builder.add_edge("general", "validate")
        
        # Validate node to END
        builder.add_edge("validate", END)
        
        # Compile graph
        return builder.compile()

    # --- NODE FUNCTIONS ---
    
    def _classify_query_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Deterministic keyword/semantic classifier node.
        Sets the query_type.
        """
        query = state["query"].lower().strip("?,.!-()\"'")
        
        # Check greetings and general help
        greetings = ["hi", "hello", "hey", "hii", "helloo", "greetings", "yo", "good morning", "good afternoon", "good evening"]
        general_keywords = ["help", "what can you do", "who are you", "what is your name", "capabilities", "how to use", "how do i use", "menu"]
        
        is_greeting = query in greetings or any(query.startswith(g + " ") for g in greetings)
        is_general = any(kw in query for kw in general_keywords) or (len(query.split()) <= 1 and query not in ["paracetamol", "aspirin", "insulin", "ibuprofen", "metformin", "amoxicillin"])
        
        if is_greeting or is_general:
            query_type = "general"
        else:
            # Check interaction keywords
            interaction_keywords = ["interact", "interaction", "together", "contraindication", "mixing", "mix", "coadministration"]
            # Check alternative keywords
            alternative_keywords = ["alternative", "alternatives", "instead of", "replace", "substitute", "other option", "options"]
            
            query_type = "drug_info" # default
            
            if any(kw in query for kw in interaction_keywords):
                query_type = "interaction"
            elif any(kw in query for kw in alternative_keywords):
                query_type = "alternative"
            
        logger.info(f"Classified query type as: '{query_type}'")
        return {"query_type": query_type}

    def _route_query(self, state: AgentState) -> str:
        """
        Determines which worker node to execute based on state query_type.
        """
        return state["query_type"]

    def _drug_info_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Worker node: calls general drug info agent.
        """
        return self.drug_agent.process(state)

    def _interaction_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Worker node: calls interaction checker agent.
        """
        return self.interaction_agent.process(state)

    def _alternative_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Worker node: calls alternative medicine agent.
        """
        return self.alternative_agent.process(state)

    def _general_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Worker node: handles greetings and general queries.
        """
        query = state["query"]
        logger.info(f"Supervisor handling general/greeting query: '{query}'")
        
        system_prompt = (
            "You are PharmaAssist DSLM, an enterprise-grade professional healthcare language assistant designed for pharmacists and medical staff.\n"
            "Greet the user politely and explain your capabilities clearly using bullet points:\n"
            "- **Clinical Chat Assistant**: Retrieve indications, dosages, warnings, and adverse reactions for essential drugs.\n"
            "- **Drug-to-Drug Interaction Checker**: Analyze interactions, contraindications, and risks between multiple medicines.\n"
            "- **WHO Essential Medicines lookup**: Check if a drug is listed in the WHO EML and find therapeutic alternatives.\n"
            "- **Vector Store Explorer**: Directly search semantic payloads in the remote Qdrant database.\n\n"
            "Be professional, concise, and helpful. Do not mention any clinical recommendations or cite any documents since this is a general greeting."
        )
        
        user_prompt = f"User greeting/query: '{query}'. Please respond appropriately."
        
        try:
            response = self.pipeline.client.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7
            )
        except Exception as e:
            response = "Hello! I am PharmaAssist DSLM, your clinical support assistant. How can I help you today?"
            
        return {
            "agent_response": response,
            "verified_sources": [],
            "citations_validated": True
        }

    def _validate_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Performs final citation checks and formatting.
        """
        logger.info("LangGraph Validation node verifying citations...")
        response = state.get("agent_response", "")
        sources = state.get("verified_sources", [])
        query_type = state.get("query_type", "")
        
        # Greetings and general queries do not require verification warnings
        if query_type == "general":
            return {"final_answer": response}
            
        # If the LLM didn't return any sources but facts were stated, we check
        # if a fallback warning should be appended
        citations_validated = state.get("citations_validated", False)
        
        if response and not sources and "not found" not in response.lower() and "no relevant" not in response.lower():
            # If the LLM answered but no sources were verified
            formatted_answer = response + "\n\n*Note: This information could not be programmatically verified against the local label databases. Please consult a healthcare professional.*"
            return {"final_answer": formatted_answer}
            
        return {"final_answer": response}

    # --- PUBLIC API ---
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Runs the full LangGraph multi-agent workflow.
        """
        logger.info(f"Starting LangGraph execution for query: '{query}'")
        initial_state = {
            "query": query,
            "query_type": "",
            "agent_response": "",
            "verified_sources": [],
            "citations_validated": False,
            "final_answer": ""
        }
        
        # Invoke LangGraph
        result = self.workflow.invoke(initial_state)
        return {
            "query_type": result.get("query_type"),
            "answer": result.get("final_answer"),
            "sources": result.get("verified_sources", []),
            "citations_validated": result.get("citations_validated", False)
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    supervisor = SupervisorAgent()
    
    # 1. Test drug info
    print("\n--- Test Query 1: Drug Info ---")
    res1 = supervisor.run("What are the side effects of Amoxicillin?")
    print("Type:", res1["query_type"])
    print("Answer:", res1["answer"][:100] + "...")
    print("Sources:", [s["brand_name"] for s in res1["sources"]])
    
    # 2. Test drug alternatives
    print("\n--- Test Query 2: Alternatives ---")
    res2 = supervisor.run("What is an alternative medicine for Amoxicillin?")
    print("Type:", res2["query_type"])
    print("Answer:", res2["answer"][:150] + "...")
    print("Sources:", [s["brand_name"] for s in res2["sources"]])
    
    # 3. Test drug interactions
    print("\n--- Test Query 3: Interactions ---")
    res3 = supervisor.run("Does Amoxicillin interact with Metformin?")
    print("Type:", res3["query_type"])
    print("Answer:", res3["answer"][:100] + "...")
    
    # 4. Test general query
    print("\n--- Test Query 4: Greetings ---")
    res4 = supervisor.run("hii")
    print("Type:", res4["query_type"])
    print("Answer:", res4["answer"])
