import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self, host: str = None, port: int = 6333, path: str = None, collection_name: str = "pharmaassist_medical"):
        self.collection_name = collection_name
        
        # Check environment variables first
        env_host = os.getenv("QDRANT_HOST")
        env_api_key = os.getenv("QDRANT_API_KEY")
        
        if env_host:
            logger.info(f"Connecting to Qdrant Cloud / Server at {env_host} (timeout=60)...")
            # If it's a cloud endpoint, use url and api_key
            if "cloud.qdrant.io" in env_host or env_api_key:
                self.client = QdrantClient(url=env_host, api_key=env_api_key, timeout=60.0)
            else:
                self.client = QdrantClient(url=env_host, timeout=60.0)
        elif host:
            logger.info(f"Connecting to Qdrant server at {host}:{port} (timeout=60)...")
            self.client = QdrantClient(host=host, port=port, timeout=60.0)
        else:
            # If no host is provided, default to a local disk storage database in the workspace
            if path is None:
                path = r"C:\Users\HP\projects\DSLM_Medical\qdrant_db"
            logger.info(f"Initializing local on-disk Qdrant storage at {path}...")
            # Ensure parent folder exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.client = QdrantClient(path=path)
            
        self.create_collection_if_not_exists()

    def create_collection_if_not_exists(self, vector_size: int = 384):
        """
        Creates the collection with HNSW indexing and Cosine distance.
        Also configures full-text search indexes on the payload.
        """
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                logger.info(f"Creating collection '{self.collection_name}' with vector size {vector_size}...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    hnsw_config=models.HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10000
                    )
                )
                
                # Create text index on brand_name and generic_name to allow fast, exact keyword filtering/matching
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="brand_name",
                    field_schema=models.TextIndexParams(
                        type="text",
                        tokenizer=models.TokenizerType.WORD,
                        min_token_len=2,
                        max_token_len=20,
                        lowercase=True
                    )
                )
                
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="generic_name",
                    field_schema=models.TextIndexParams(
                        type="text",
                        tokenizer=models.TokenizerType.WORD,
                        min_token_len=2,
                        max_token_len=20,
                        lowercase=True
                    )
                )
                
                # Create payload index for sections and essential status
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="section",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="is_essential",
                    field_schema=models.PayloadSchemaType.BOOL
                )
                
                logger.info(f"Collection '{self.collection_name}' created and indexed successfully.")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Upserts a batch of chunks into Qdrant.
        """
        if not chunks or not embeddings:
            return
            
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            chunk_id = chunk["chunk_id"]
            # Convert a string ID (like 'spl_set_id_section_idx') to a UUID if not a valid Qdrant ID.
            # Qdrant client handles string UUIDs, integer IDs. If our chunk ID is a string, we can use it directly,
            # or hash it to a UUID to ensure compatibility.
            import uuid
            # Qdrant supports UUID strings directly. Let's create a deterministic UUID based on our string chunk_id.
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            
            payload = {
                "text": chunk["text"],
                "raw_content": chunk["raw_content"],
                **chunk["metadata"]
            }
            
            points.append(PointStruct(
                id=point_uuid,
                vector=vector,
                payload=payload
            ))
            
        import time
        max_attempts = 3
        backoff = 1.0
        for attempt in range(max_attempts):
            try:
                logger.info(f"Upserting {len(points)} points to collection '{self.collection_name}' (attempt {attempt + 1}/{max_attempts})...")
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info("Upsert successful.")
                return
            except Exception as e:
                logger.error(f"Error upserting points to Qdrant on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    logger.error("All upsert attempts failed.")
                    raise e

    def semantic_search(self, query_vector: List[float], limit: int = 10, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search with retry logic.
        """
        import time
        qdrant_filter = None
        if filter_dict:
            conditions = []
            for k, v in filter_dict.items():
                conditions.append(models.FieldCondition(
                    key=k,
                    match=models.MatchValue(value=v)
                ))
            qdrant_filter = models.Filter(must=conditions)
            
        max_attempts = 3
        backoff = 1.0
        for attempt in range(max_attempts):
            try:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=qdrant_filter,
                    limit=limit
                )
                results = response.points
                
                formatted_results = []
                for res in results:
                    formatted_results.append({
                        "id": res.id,
                        "score": res.score,
                        "text": res.payload.get("text", ""),
                        "raw_content": res.payload.get("raw_content", ""),
                        "metadata": {k: v for k, v in res.payload.items() if k not in ["text", "raw_content"]}
                    })
                return formatted_results
            except Exception as e:
                logger.error(f"Error executing semantic search in Qdrant (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    return []

    def keyword_search(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Performs keyword match search on brand_name and generic_name payload fields with retry logic.
        """
        import time
        max_attempts = 3
        backoff = 1.0
        for attempt in range(max_attempts):
            try:
                # Query brand name or generic name matches
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        should=[
                            models.FieldCondition(
                                key="brand_name",
                                match=models.MatchText(text=query_text)
                            ),
                            models.FieldCondition(
                                key="generic_name",
                                match=models.MatchText(text=query_text)
                            )
                        ]
                    ),
                    limit=limit,
                    with_payload=True
                )[0]
                
                formatted_results = []
                for res in results:
                    formatted_results.append({
                        "id": res.id,
                        # Scroll results don't have a similarity score, we assign a nominal match score
                        "score": 1.0, 
                        "text": res.payload.get("text", ""),
                        "raw_content": res.payload.get("raw_content", ""),
                        "metadata": {k: v for k, v in res.payload.items() if k not in ["text", "raw_content"]}
                    })
                return formatted_results
            except Exception as e:
                logger.error(f"Error executing keyword search in Qdrant (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = QdrantManager(collection_name="test_collection")
    # Quick test upsert
    test_chunks = [{
        "chunk_id": "test_1",
        "text": "Drug: Metformin. Section: Indications. Content: Used for diabetes.",
        "raw_content": "Used for diabetes.",
        "metadata": {"brand_name": "Glucophage", "generic_name": "Metformin", "section": "indications"}
    }]
    test_embs = [[0.1] * 384]
    manager.upsert_chunks(test_chunks, test_embs)
    
    # Quick search
    res = manager.semantic_search(query_vector=[0.1] * 384, limit=1)
    print("Search result:", res)
