import os
import time
import logging
from typing import List, Dict, Any
from ingestion.who_loader import WHOLoader
from ingestion.openfda_loader import OpenFDALoader
from chunking.splitter import MedicalChunkSplitter
from embeddings.embedder import MedicalEmbedder
from vectorstore.qdrant_manager import QdrantManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingestion_pipeline")

class IngestionPipeline:
    def __init__(self, 
                 workspace_path: str = r"C:\Users\HP\projects\DSLM_Medical",
                 collection_name: str = "pharmaassist_medical",
                 batch_size: int = 100):
        self.workspace_path = workspace_path
        self.collection_name = collection_name
        self.batch_size = batch_size
        
        # Initialize components
        self.who_loader = WHOLoader(os.path.join(workspace_path, "eml_export.xlsx"))
        self.fda_loader = OpenFDALoader(workspace_path)
        self.splitter = MedicalChunkSplitter(chunk_size=800, chunk_overlap=150)
        self.embedder = MedicalEmbedder(model_name="BAAI/bge-small-en-v1.5", device="cpu")
        self.qdrant = QdrantManager(collection_name=collection_name, path=os.path.join(workspace_path, "qdrant_db"))

    def run(self, max_records_per_file: int = -1, only_essential: bool = True):
        """
        Runs the ingestion pipeline.
        max_records_per_file: Limit records parsed per zip file (useful for fast testing).
        only_essential: If True, only ingests drugs listed in the WHO Essential Medicines List.
        """
        start_time = time.time()
        logger.info(f"Starting ingestion pipeline. Filters: only_essential={only_essential}, max_records_per_file={max_records_per_file}")
        
        # Generator for streaming cleaned records
        records_stream = self.fda_loader.stream_records(
            max_records_per_file=max_records_per_file,
            only_essential=only_essential,
            who_loader=self.who_loader
        )
        
        chunk_batch = []
        total_processed_records = 0
        total_indexed_chunks = 0
        
        for record in records_stream:
            total_processed_records += 1
            
            # Split record into chunks
            record_chunks = self.splitter.split_record(record)
            if not record_chunks:
                continue
                
            chunk_batch.extend(record_chunks)
            
            # When batch size is met, process and insert
            if len(chunk_batch) >= self.batch_size:
                self._process_and_insert_batch(chunk_batch)
                total_indexed_chunks += len(chunk_batch)
                chunk_batch = []
                
        # Process remaining chunks in the buffer
        if chunk_batch:
            self._process_and_insert_batch(chunk_batch)
            total_indexed_chunks += len(chunk_batch)
            
        elapsed = time.time() - start_time
        logger.info("==================================================")
        logger.info("Ingestion Pipeline Completion Summary:")
        logger.info(f"  Total Processed Drugs: {total_processed_records}")
        logger.info(f"  Total Indexed Chunks:  {total_indexed_chunks}")
        logger.info(f"  Time Elapsed:          {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        logger.info("==================================================")
        
        return total_processed_records, total_indexed_chunks

    def _process_and_insert_batch(self, chunks: List[Dict[str, Any]]):
        """
        Embeds a list of chunks and inserts them into Qdrant.
        """
        logger.info(f"Processing batch of {len(chunks)} chunks (generating embeddings)...")
        # Extract the text for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings on CPU
        embeddings = self.embedder.embed_documents(texts)
        
        # Upsert into Qdrant
        logger.info(f"Upserting {len(chunks)} chunks into vector store...")
        self.qdrant.upsert_chunks(chunks, embeddings)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    # Read configuration limits from environment variables
    max_records = int(os.getenv("MAX_RECORDS_PER_FILE", "50"))
    only_essential = os.getenv("ONLY_ESSENTIAL_DRUGS", "True").lower() == "true"
    
    pipeline = IngestionPipeline(batch_size=30)
    pipeline.run(max_records_per_file=max_records, only_essential=only_essential)
