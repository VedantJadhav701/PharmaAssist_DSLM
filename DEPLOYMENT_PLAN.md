# DEPLOYMENT_PLAN.md

# PharmaAssist DSLM Deployment & Production Scale-up Plan

This document outlines the roadmap, architecture configuration, and system orchestration requirements for moving the **PharmaAssist DSLM** system from local development to an enterprise-grade production environment.

---

## 1. Production Architecture (Containerized)

For a robust cloud or on-premise production deployment, we containerize all modules using Docker and coordinate them via `docker-compose`.

```mermaid
graph TD
    User([Client / Web Browser]) -->|HTTPS| Nginx[NGINX Reverse Proxy]
    Nginx -->|Port 8501| Streamlit[Streamlit Container]
    Nginx -->|Port 8000| FastAPI[FastAPI App Containers]
    
    subgraph FastAPI Service Cluster
        FastAPI -->|Load Balancing| Gunicorn[Gunicorn / Uvicorn Workers]
    end
    
    subgraph Vector Store Database
        Gunicorn -->|REST API Port 443| QdrantCloud[(Managed Qdrant Cloud)]
    end
    
    subgraph Inference Cluster
        Gunicorn -->|API Requests| OllamaServer[Ollama Server / vLLM Instance]
        OllamaServer -->|Inference GPU| GPUCluster[NVIDIA GPU A10G / A100]
    end
```

### Docker Compose Configuration (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant_local_fallback
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: always
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

  fastapi_backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: fastapi_backend
    ports:
      - "8000:8000"
    environment:
      - QDRANT_HOST=https://9f4082d7-e6c8-4329-aef6-db592a9d874c.eu-west-2-0.aws.cloud.qdrant.io
      - QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6M2IwNTEyNmMtMmI3MS00NTNhLWE3ZmItMTQyODdkZjZjNDgyIn0.aBpNO7ux-4O4itbdVeZE9Py3PHYXdRK3nw2UMhMFwXk
      - OLLAMA_HOST=http://ollama_inference:11434
      - OLLAMA_MODEL=qwen2.5:1.5b-instruct-q4_K_M
    depends_on:
      - qdrant
    restart: always

  streamlit_frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    container_name: streamlit_frontend
    ports:
      - "8501:8501"
    environment:
      - BACKEND_API_URL=http://fastapi_backend:8000/api
    depends_on:
      - fastapi_backend
    restart: always

volumes:
  qdrant_data:
```

---

## 2. Infrastructure & Sizing Guidelines

Depending on user volume and hardware capabilities, choose one of the following two deployment tiers:

### Tier 1: Local / On-Premise Clinic Deployment
* **Purpose**: Small pharmacy, local hospital clinic, or standalone medical store.
* **Hardware Sizing**:
  * **GPU**: NVIDIA RTX 3060 / 4060 (8GB VRAM) or RTX 3050 (4GB/8GB).
  * **RAM**: 16GB System RAM.
  * **CPU**: Intel Core i5/i7 or AMD Ryzen 5/7 (6+ cores).
* **Software Configuration**:
  * Ollama running Qwen2.5-1.5B (or Qwen2.5-3B for enhanced reasoning).
  * Embedding and Reranker models running on CPU (SentenceTransformers).
  * Qdrant Cloud or local SQLite-based Qdrant on local SSD.

### Tier 2: Cloud-Scale Enterprise Deployment
* **Purpose**: National pharmacy chain, multi-department hospital network, or medical search portal.
* **Hardware Sizing**:
  * **Inference Host**: AWS EC2 instance `g5.xlarge` (1x NVIDIA A10G GPU, 24GB VRAM) or Google Cloud `g2-standard-4`.
  * **Database**: Managed Qdrant Cloud Cluster (Tier-1 or Tier-2 Cluster with automatic scaling and backup).
  * **API App Host**: AWS ECS/Fargate or Kubernetes Cluster (EKS) for auto-scaled FastAPI workers.
* **Software Configuration**:
  * vLLM or HuggingFace TGI (Text Generation Inference) running Qwen2.5-7B-Instruct or Qwen2.5-14B-Instruct for supreme clinical reasoning.
  * Embeddings and Reranker served via Text Embeddings Inference (TEI) containers for sub-millisecond latencies.

---

## 3. Large-Scale Ingestion Plan (Full 15GB OpenFDA Dataset)

Processing the entire OpenFDA dataset (13 zipped files, expanding to ~15GB of raw JSON text) on a single thread will crash or take days. We use a **distributive, queue-based streaming worker** architecture:

```text
               OpenFDA ZIP Files (S3 bucket)
                            │
                            ▼
                   Celery Beat Scheduler
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
         Celery Task 1             Celery Task 2      ... (Celery Task N)
       [Read File 0001]          [Read File 0002]
               │                         │
               ▼                         ▼
         Streaming Parser          Streaming Parser
         & WHO Filter              & WHO Filter
               │                         │
               └────────────┬────────────┘
                            ▼
                     Redis Message Broker
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
         GPU Worker A              GPU Worker B
       [Vector Embedding]        [Vector Embedding]
               │                         │
               └────────────┬────────────┘
                            ▼
                   Qdrant Cloud API
```

1. **Storage**: Store the raw ZIP files on a cloud object store (e.g. AWS S3 or Google Cloud Storage).
2. **Task Queue**: Use **Celery** with a **Redis** or **RabbitMQ** message broker.
3. **Chunking Workers**: Worker processes stream records from S3, parse using `ijson` in chunks, filter against WHO EML, split sections, and queue the chunks in batches of 50.
4. **Embedding Workers**: Batch embedding workers retrieve chunk lists, send them to a dedicated GPU inference pool (e.g. Triton Inference Server or TEI), and bulk upsert them into Qdrant Cloud.
5. **Deduplication**: Use Qdrant's `upsert` with deterministic UUIDs based on the SPL Set ID to prevent duplicates if ingestion is run multiple times.

---

## 4. Security & Compliance (HIPAA / GDPR)

Healthcare data requires strict compliance:
1. **Zero-PII Storage**: The vector store and logs must never capture Patient Identifiable Information (PII). All user-queries are filtered on the FastAPI backend using PII scrubbing libraries (e.g., Microsoft Presidio) before being sent to the retriever.
2. **Local Inference Isolation**: For high-compliance clinic environments, keep Ollama and Qdrant local (Tier 1) inside the hospital's private intranet. Block external HTTP/HTTPS egress.
3. **Transit Encryption**: Force TLS 1.3/HTTPS for all connections between the Streamlit client, FastAPI API, and Qdrant Cloud.
4. **Audit Logs**: Store all executed queries, latencies, and citation validation reports in an encrypted log store for compliance auditing.
