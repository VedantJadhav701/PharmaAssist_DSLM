# PharmaAssist DSLM - Project Context

## Project Overview

PharmaAssist DSLM is an enterprise-grade Domain Specific Language Model (DSLM) designed for pharmacists, hospitals, healthcare professionals, and medical stores.

The objective is to provide accurate, evidence-backed, and citation-based responses for drug-related queries using Retrieval Augmented Generation (RAG), domain-specific knowledge, and agentic workflows.

The system prioritizes:

* Accuracy
* Explainability
* Traceability
* Low Hallucination
* Healthcare Compliance

---

# Problem Statement

Healthcare professionals spend significant time searching through:

* Drug labels
* Dosage guidelines
* Contraindications
* Adverse reactions
* Drug interactions
* Prescribing information

This information is scattered across multiple sources.

PharmaAssist DSLM centralizes and retrieves this knowledge using semantic search and LLM reasoning.

---

# Target Users

## Primary Users

* Pharmacists
* Chemists
* Medical Stores

## Secondary Users

* Hospitals
* Healthcare Staff
* Medical Researchers

---

# Data Sources

## Local Knowledge Base

### OpenFDA Drug Labels

Downloaded Dataset:

```text
drug-label-0001-of-0013.json.zip
...
drug-label-0013-of-0013.json.zip
```

Contains:

* Drug Names
* Usage
* Warnings
* Dosage
* Contraindications
* Adverse Reactions
* Interactions

---

### WHO Essential Medicines

File:

```text
eml_export.xlsx
```

Contains:

* Essential Medicines
* Drug Categories
* Medical Classifications
* Therapeutic Groups

Used primarily as metadata enrichment.

---

# Live Knowledge Sources

## DailyMed API

Used for:

* Latest Prescribing Information
* Updated Labels
* Drug Safety Updates

Purpose:

Avoid stale information while keeping local storage manageable.

---

# System Architecture

```text
                    User
                      │
                      ▼

              Streamlit UI

                      │
                      ▼

                FastAPI API

                      │
                      ▼

              Query Router

                      │
      ┌───────────────┼────────────────┐
      │               │                │

      ▼               ▼                ▼

 Drug Info      Interaction      Document QA
   Agent           Agent            Agent

      └───────────────┼────────────────┘
                      ▼

              Retrieval Layer

                      ▼

                  Qdrant

                      ▲

                Embeddings

                      ▲

                Chunk Store

                      ▲

      ┌───────────────┼────────────────┐
      │               │                │

   OpenFDA         WHO EML       DailyMed API

                      ▼

                 Qwen 2.5

                      ▼

               Citation Layer

                      ▼

                Final Answer
```

---

# Technology Stack

## Frontend

Streamlit

Purpose:

* Chat Interface
* Document Upload
* Drug Search
* Citation Display

---

## Backend

FastAPI

Purpose:

* API Layer
* Agent Orchestration
* Retrieval Calls

---

## Embeddings

Model:

BAAI/bge-small-en-v1.5

Purpose:

Convert medical text into vectors.

---

## Vector Database

Qdrant

Purpose:

Store and retrieve semantic medical knowledge.

---

## LLM

Qwen 2.5 1.5B Instruct

Running locally through Ollama.

Purpose:

Reasoning and response generation.

---

## Agent Framework

LangGraph

Purpose:

Multi-agent orchestration.

---

# Data Pipeline

```text
OpenFDA JSON
        ↓
Data Cleaning
        ↓
Metadata Extraction
        ↓
Chunking
        ↓
Embedding Generation
        ↓
Qdrant Storage
```

---

# Metadata Schema

```json
{
  "drug_name": "",
  "source": "",
  "indications": "",
  "warnings": "",
  "dosage": "",
  "contraindications": "",
  "adverse_reactions": "",
  "category": ""
}
```

---

# Phase 1

Objectives:

* OpenFDA ingestion
* WHO integration
* Embedding generation
* Qdrant indexing
* Basic RAG chatbot

Deliverable:

Drug information assistant with citations.

---

# Phase 2

Objectives:

* Drug interaction detection
* Alternative medicine suggestions
* Dosage lookup

Deliverable:

Pharmacist Assistant.

---

# Phase 3

Objectives:

* DailyMed integration
* Live medical retrieval
* Hybrid RAG

Deliverable:

Real-time Healthcare DSLM.

---

# Phase 4

Objectives:

* Multi-Agent System

Agents:

1. Drug Agent
2. Interaction Agent
3. Research Agent
4. Safety Agent

Deliverable:

Agentic Healthcare DSLM.

---

# Evaluation Metrics

Use:

* RAGAS
* Faithfulness
* Answer Relevance
* Context Precision
* Context Recall

Goal:

Minimize hallucinations while maximizing factual correctness.

---

# Success Criteria

The system should:

* Answer drug-related questions accurately.
* Provide source citations.
* Detect interactions.
* Support semantic medical search.
* Run efficiently on RTX 3050 4GB hardware.
* Be deployable locally or on cloud infrastructure.

```
```
