import logging
import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from agents.supervisor_agent import SupervisorAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api_main")

app = FastAPI(
    title="PharmaAssist DSLM API",
    description="Enterprise-grade Healthcare Domain-Specific Language Model backend for pharmaceutical info.",
    version="1.0"
)

# CORS configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Supervisor Agent instance (Singleton/Dependency Injection pattern)
workspace_path = r"C:\Users\HP\projects\DSLM_Medical"
supervisor = None

@app.on_event("startup")
def startup_event():
    global supervisor
    logger.info("Starting up PharmaAssist DSLM backend...")
    try:
        supervisor = SupervisorAgent(workspace_path=workspace_path)
        logger.info("Supervisor Agent and local models initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supervisor Agent: {e}")
        # We don't crash startup, but endpoints will report 503 if not ready

# --- Request / Response Schemas ---

class ChatRequest(BaseModel):
    query: str = Field(..., example="What are the warnings for Amoxicillin?")

class ChatResponse(BaseModel):
    query_type: str = Field(..., description="Classified type of query (drug_info, interaction, alternative)")
    answer: str = Field(..., description="Generated answer from the model")
    sources: List[Dict[str, Any]] = Field(default=[], description="List of verified and structured document sources cited")
    success: bool = Field(True, description="Indicating successful completion")

class InteractionRequest(BaseModel):
    drugs: List[str] = Field(..., min_items=2, example=["Amoxicillin", "Metformin"])

class DrugSearchRequest(BaseModel):
    drug_name: str = Field(..., example="Amoxicillin")

# --- Endpoints ---

@app.get("/api/health")
def health_check():
    """
    Returns health status of API and Qdrant/Ollama connectivity.
    """
    is_ready = supervisor is not None
    return {
        "status": "healthy" if is_ready else "uninitialized",
        "api_version": "1.0",
        "database": "connected" if is_ready else "offline",
        "ollama": "connected" if is_ready else "offline"
    }

@app.post("/api/chat", response_model=ChatResponse)
def post_chat(request: ChatRequest):
    """
    Unified chat endpoint. Classifies, routes, and answers drug queries using LangGraph.
    """
    if supervisor is None:
        raise HTTPException(status_code=503, detail="Supervisor Agent is not initialized.")
        
    try:
        logger.info(f"API received query: '{request.query}'")
        res = supervisor.run(request.query)
        return ChatResponse(
            query_type=res["query_type"],
            answer=res["answer"],
            sources=res["sources"],
            success=True
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/interactions", response_model=ChatResponse)
def post_interactions(request: InteractionRequest):
    """
    Dedicated drug-to-drug interaction checker.
    """
    if supervisor is None:
        raise HTTPException(status_code=503, detail="Supervisor Agent is not initialized.")
        
    try:
        query = f"Does {' and '.join(request.drugs)} interact?"
        logger.info(f"API received interaction request for: {request.drugs}")
        res = supervisor.run(query)
        return ChatResponse(
            query_type="interaction",
            answer=res["answer"],
            sources=res["sources"],
            success=True
        )
    except Exception as e:
        logger.error(f"Error in interaction endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drug", response_model=ChatResponse)
def post_drug(request: DrugSearchRequest):
    """
    Dedicated drug information search.
    """
    if supervisor is None:
        raise HTTPException(status_code=503, detail="Supervisor Agent is not initialized.")
        
    try:
        query = f"What is the dosage, warnings, and indications for {request.drug_name}?"
        logger.info(f"API received drug lookup for: {request.drug_name}")
        res = supervisor.run(query)
        return ChatResponse(
            query_type="drug_info",
            answer=res["answer"],
            sources=res["sources"],
            success=True
        )
    except Exception as e:
        logger.error(f"Error in drug search endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Runs the uvicorn server locally on port 8000
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
