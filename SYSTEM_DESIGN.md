# SYSTEM_DESIGN.md

# PharmaAssist DSLM System Design

Version: 1.0

Author: Vedant Jadhav

Project: PharmaAssist DSLM

---

# High-Level Goal

Build a Healthcare Domain-Specific Language Model (DSLM) capable of:

* Drug Information Retrieval
* Drug Interaction Detection
* Dosage Guidance
* Contraindication Detection
* Alternative Medicine Suggestions
* Citation-Based Responses

while running efficiently on local hardware.

Target Hardware:

RTX 3050 4GB

---

# Repository Structure

PharmaAssist_DSLM/

├── data/

│   ├── openfda/

│   ├── who/

│   ├── processed/

│

├── ingestion/

│   ├── openfda_loader.py

│   ├── who_loader.py

│   ├── cleaner.py

│

├── chunking/

│   ├── splitter.py

│

├── embeddings/

│   ├── embedder.py

│

├── vectorstore/

│   ├── qdrant_manager.py

│

├── retrieval/

│   ├── retriever.py

│   ├── reranker.py

│

├── agents/

│   ├── supervisor_agent.py

│   ├── drug_agent.py

│   ├── interaction_agent.py

│   ├── research_agent.py

│

├── llm/

│   ├── qwen_client.py

│

├── api/

│   ├── main.py

│

├── frontend/

│   ├── streamlit_app.py

│

├── evaluation/

│   ├── ragas_eval.py

│

└── docs/

```
├── PROJECT_CONTEXT.md

├── SYSTEM_DESIGN.md
```

---

# Data Architecture

## OpenFDA

Source:

drug-label-0001-of-0013.json.zip

to

drug-label-0013-of-0013.json.zip

Fields to Extract:

* brand_name
* generic_name
* indications_and_usage
* dosage_and_administration
* warnings
* contraindications
* adverse_reactions
* drug_interactions

---

## WHO Essential Medicines

Source:

eml_export.xlsx

Fields:

* Medicine Name
* Indication
* Category
* Therapeutic Group

Purpose:

Metadata enrichment.

---

# Data Flow

OpenFDA JSON

↓

Parser

↓

Cleaner

↓

Metadata Builder

↓

Chunking

↓

Embedding Generation

↓

Qdrant Storage

↓

Retriever

↓

LLM

↓

Answer

---

# Chunking Strategy

Method:

Recursive Character Text Splitter

Configuration:

Chunk Size: 800

Chunk Overlap: 150

Reason:

Maintains medical context while improving retrieval.

Example:

Chunk 1

Metformin indications...

Chunk 2

Metformin warnings...

Chunk 3

Metformin interactions...

---

# Metadata Schema

{
"drug_name": "",
"generic_name": "",
"category": "",
"source": "",
"section": "",
"chunk_id": ""
}

---

# Embedding Layer

Model:

BAAI/bge-small-en-v1.5

Dimension:

384

Reason:

* Lightweight
* High retrieval quality
* Works efficiently on CPU

Pipeline:

Text

↓

Embedding

↓

Vector

↓

Qdrant

---

# Vector Database

Database:

Qdrant

Collection:

pharmaassist_medical

Stored Data:

* Embeddings
* Metadata
* Chunk Text

Index Type:

HNSW

Distance Metric:

Cosine Similarity

---

# Retrieval Layer

Input:

User Query

Example:

Can Metformin interact with Ibuprofen?

Pipeline:

Query

↓

Embedding

↓

Vector Search

↓

Top 10 Results

↓

Reranker

↓

Top 5 Results

---

# Reranking Layer

Model:

BAAI/bge-reranker-base

Purpose:

Improve relevance before sending context to LLM.

Pipeline:

Top 10 Chunks

↓

Reranker

↓

Top 5 Chunks

↓

LLM

---

# LLM Layer

Model:

Qwen2.5-1.5B-Instruct

Runtime:

Ollama

Responsibilities:

* Medical reasoning
* Summarization
* Citation generation
* Response formatting

Not Responsible For:

* Diagnosis
* Prescription

---

# Prompt Template

SYSTEM:

You are PharmaAssist DSLM, a healthcare domain assistant.

Only answer using the retrieved medical context.

If the information is unavailable, explicitly state that the information was not found.

Never invent medical facts.

Always provide citations.

USER:

{question}

CONTEXT:

{retrieved_chunks}

---

# Agent Architecture

Supervisor Agent

↓

Classify Query

↓

Drug Information?

↓

Drug Agent

↓

Interaction Query?

↓

Interaction Agent

↓

Research Query?

↓

Research Agent

↓

Final Response

---

# LangGraph Workflow

START

↓

Query Classification

↓

Retrieve Context

↓

Agent Selection

↓

Generate Response

↓

Citation Validation

↓

END

---

# FastAPI Endpoints

POST

/api/chat

Purpose:

Medical Question Answering

---

POST

/api/interactions

Purpose:

Drug Interaction Lookup

---

POST

/api/drug

Purpose:

Drug Information Search

---

POST

/api/upload

Purpose:

Future PDF Upload Support

---

# Streamlit Pages

Page 1

Dashboard

Features:

* Search Bar
* Recent Queries

---

Page 2

Drug Search

Features:

* Drug Lookup
* Dosage
* Warnings

---

Page 3

Interaction Checker

Input:

Drug A

Drug B

Output:

Interaction Analysis

---

Page 4

Sources Explorer

Features:

* View Retrieved Chunks
* View Citations

---

# Evaluation

Framework:

RAGAS

Metrics:

Faithfulness

Answer Relevance

Context Precision

Context Recall

Hallucination Rate

---

# Version Roadmap

V1

OpenFDA RAG

WHO Metadata

Qdrant

Qwen

Drug Information

---

V2

Drug Interactions

Alternative Medicines

Citation Validation

---

V3

DailyMed Integration

Live Retrieval

Hybrid Knowledge System

---

V4

Multi-Agent Healthcare DSLM

LangGraph

Research Agent

Safety Agent

Supervisor Agent

---

# Production Goal

Enterprise Healthcare DSLM capable of providing accurate, citation-backed pharmaceutical information with low hallucination rates while operating efficiently on commodity hardware.
