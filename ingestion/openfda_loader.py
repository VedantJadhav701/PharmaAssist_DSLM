import os
import zipfile
import json
import logging
from typing import Generator, Dict, Any, List
from ingestion.cleaner import clean_drug_field, extract_names
from ingestion.who_loader import WHOLoader

logger = logging.getLogger(__name__)

class OpenFDALoader:
    def __init__(self, workspace_path: str = r"C:\Users\HP\projects\DSLM_Medical"):
        self.workspace_path = workspace_path
        self.zip_files = self._discover_zip_files()
        
    def _discover_zip_files(self) -> List[str]:
        """
        Discovers all drug-label ZIP files in the workspace.
        """
        zip_files = []
        if os.path.exists(self.workspace_path):
            for file in os.listdir(self.workspace_path):
                if file.startswith("drug-label-") and file.endswith(".json.zip"):
                    zip_files.append(os.path.join(self.workspace_path, file))
        
        # Sort them numerically
        zip_files.sort()
        logger.info(f"Discovered {len(zip_files)} OpenFDA ZIP files.")
        return zip_files

    def stream_records(self, max_records_per_file: int = -1, only_essential: bool = True, who_loader: WHOLoader = None) -> Generator[Dict[str, Any], None, None]:
        """
        Streams, cleans, and filters records from the ZIP files sequentially.
        """
        if not who_loader:
            logger.info("Initializing WHO EML Loader for filtering...")
            who_loader = WHOLoader()
            
        for zip_path in self.zip_files:
            logger.info(f"Streaming records from {os.path.basename(zip_path)}...")
            yield from self._stream_zip_file(zip_path, max_records_per_file, only_essential, who_loader)

    def _stream_zip_file(self, zip_path: str, max_records: int, only_essential: bool, who_loader: WHOLoader) -> Generator[Dict[str, Any], None, None]:
        """
        Helper to parse a single ZIP file record-by-record using low-RAM streaming.
        """
        try:
            count = 0
            with zipfile.ZipFile(zip_path, 'r') as z:
                for name in z.namelist():
                    if not name.endswith('.json'):
                        continue
                    
                    with z.open(name) as f:
                        # Standard stream reading logic
                        # Since we want to parse it without loading the whole file:
                        # Read in 16MB blocks and parse using a custom braces parser for objects in the 'results' list
                        # This works because the results array contains independent JSON objects
                        buffer = ""
                        in_results = False
                        brace_count = 0
                        obj_chars = []
                        
                        while True:
                            chunk = f.read(64 * 1024) # 64KB chunks
                            if not chunk:
                                break
                            
                            text = chunk.decode('utf-8', errors='ignore')
                            
                            for char in text:
                                if not in_results:
                                    # Look for results array start
                                    buffer += char
                                    if '"results":' in buffer:
                                        # Clear buffer, wait for '['
                                        buffer = ""
                                    if len(buffer) > 100:
                                        # Keep buffer small
                                        buffer = buffer[-50:]
                                    if char == '[' and '"results"' in text[:text.find(char)]:
                                        in_results = True
                                    elif char == '[':
                                        # Check if we crossed into results
                                        in_results = True
                                    continue
                                
                                # We are in results array
                                if char == '{':
                                    brace_count += 1
                                    obj_chars.append(char)
                                elif char == '}':
                                    brace_count -= 1
                                    if obj_chars:
                                        obj_chars.append(char)
                                    
                                    if brace_count == 0 and obj_chars:
                                        # Completed an object
                                        obj_str = "".join(obj_chars)
                                        obj_chars = []
                                        
                                        try:
                                            record = json.loads(obj_str)
                                            cleaned = self._process_record(record, who_loader)
                                            if cleaned:
                                                # Check if we should filter by WHO EML
                                                is_essential = cleaned["eml_metadata"].get("is_essential", False)
                                                if not only_essential or is_essential:
                                                    yield cleaned
                                                    count += 1
                                                    if max_records > 0 and count >= max_records:
                                                        logger.info(f"Reached limit of {max_records} records for this file.")
                                                        return
                                        except Exception as e:
                                            # Skip malformed JSON objects
                                            pass
                                elif brace_count > 0:
                                    obj_chars.append(char)
        except Exception as e:
            logger.error(f"Error streaming zip file {zip_path}: {e}")

    def _process_record(self, record: Dict[str, Any], who_loader: WHOLoader) -> Dict[str, Any]:
        """
        Cleans and packages a single OpenFDA raw JSON record.
        """
        openfda = record.get("openfda", {})
        if not openfda:
            return None
        
        brand_name, generic_name = extract_names(openfda)
        if not generic_name and not brand_name:
            return None
            
        # Unique ID for deduplication
        set_id = record.get("set_id")
        if not set_id:
            set_id = openfda.get("spl_set_id", [None])[0]
        if not set_id:
            set_id = record.get("id")
        if not set_id:
            return None
            
        # Get EML metadata
        eml_meta = who_loader.get_metadata(generic_name)
        
        # Extract text sections
        indications = clean_drug_field(record.get("indications_and_usage"))
        warnings = clean_drug_field(record.get("warnings"))
        contraindications = clean_drug_field(record.get("contraindications"))
        dosage = clean_drug_field(record.get("dosage_and_administration"))
        adverse_reactions = clean_drug_field(record.get("adverse_reactions"))
        interactions = clean_drug_field(record.get("drug_interactions"))
        
        # We need at least one text section to make chunking useful
        if not (indications or warnings or contraindications or dosage or adverse_reactions or interactions):
            return None
            
        return {
            "id": set_id,
            "brand_name": brand_name,
            "generic_name": generic_name,
            "indications_and_usage": indications,
            "warnings": warnings,
            "contraindications": contraindications,
            "dosage_and_administration": dosage,
            "adverse_reactions": adverse_reactions,
            "drug_interactions": interactions,
            "eml_metadata": eml_meta
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = OpenFDALoader()
    who_loader = WHOLoader()
    
    # Quick stream test
    print("Starting sample stream (max 3 essential drugs)...")
    for rec in loader.stream_records(max_records_per_file=5, only_essential=True, who_loader=who_loader):
        print(f"Brand: {rec['brand_name']} | Generic: {rec['generic_name']} | Essential: {rec['eml_metadata'].get('is_essential')}")
        print(f"Indications (first 100 chars): {rec['indications_and_usage'][:100]}...\n")
        break
