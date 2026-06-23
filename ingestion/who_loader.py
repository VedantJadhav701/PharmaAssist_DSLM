import os
import re
import pandas as pd
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def clean_medicine_name(name: str) -> str:
    """
    Cleans a medicine name for robust matching.
    Converts to lowercase, removes salt forms (e.g., 'hydrochloride', 'sulfate', 'sodium'),
    removes parenthetical details, and strips whitespace.
    """
    if not isinstance(name, str):
        return ""
    
    # Lowercase
    name = name.lower()
    
    # Remove text in parentheses
    name = re.sub(r'\(.*?\)', '', name)
    
    # Remove common salts and terms that complicate matching
    salts = [
        r'\bhydrochloride\b', r'\bsodium\b', r'\bsulfate\b', r'\bchloride\b',
        r'\bphosphate\b', r'\bacetate\b', r'\bmesylate\b', r'\bmaleate\b',
        r'\btartrate\b', r'\bcalcium\b', r'\bpotassium\b', r'\bgluconate\b',
        r'\bfumarate\b', r'\bsuccinate\b', r'\bhydrate\b', r'\banhydrous\b'
    ]
    for salt in salts:
        name = re.sub(salt, '', name)
        
    # Clean whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name

class WHOLoader:
    def __init__(self, file_path: str = None):
        if file_path is None:
            # Default lookup paths
            paths_to_try = [
                r"C:\Users\HP\projects\DSLM_Medical\eml_export.xlsx",
                "eml_export.xlsx",
                "data/who/eml_export.xlsx"
            ]
            for p in paths_to_try:
                if os.path.exists(p):
                    file_path = p
                    break
        
        self.file_path = file_path
        self.eml_db = {}
        
        if file_path and os.path.exists(file_path):
            self.load()
        else:
            logger.warning(f"WHO EML file not found at {file_path}. Ingestion will proceed without EML metadata.")

    def load(self):
        try:
            logger.info(f"Loading WHO EML from {self.file_path}...")
            df = pd.read_excel(self.file_path)
            
            # Ensure required columns are present
            required_cols = ['Medicine name', 'EML section', 'Indication', 'ATC codes']
            for col in required_cols:
                if col not in df.columns:
                    # Fallback to check case-insensitive match
                    for actual_col in df.columns:
                        if actual_col.lower() == col.lower():
                            df.rename(columns={actual_col: col}, inplace=True)
                            break
            
            for _, row in df.iterrows():
                med_name = row.get('Medicine name')
                if pd.isna(med_name):
                    continue
                
                med_name = str(med_name).strip()
                cleaned_name = clean_medicine_name(med_name)
                if not cleaned_name:
                    continue
                
                section = row.get('EML section', 'Unknown')
                indication = row.get('Indication', 'Unknown')
                atc_codes = row.get('ATC codes', '')
                
                # A single medicine might have multiple rows (different formulations/sections)
                # Let's group them
                if cleaned_name not in self.eml_db:
                    self.eml_db[cleaned_name] = {
                        "original_name": med_name,
                        "sections": {str(section)} if pd.notna(section) else set(),
                        "indications": {str(indication)} if pd.notna(indication) else set(),
                        "atc_codes": {c.strip() for c in str(atc_codes).split(',')} if pd.notna(atc_codes) else set()
                    }
                else:
                    if pd.notna(section):
                        self.eml_db[cleaned_name]["sections"].add(str(section))
                    if pd.notna(indication):
                        self.eml_db[cleaned_name]["indications"].add(str(indication))
                    if pd.notna(atc_codes):
                        for c in str(atc_codes).split(','):
                            self.eml_db[cleaned_name]["atc_codes"].add(c.strip())
            
            # Convert sets to sorted lists for JSON serialization later
            for name in self.eml_db:
                self.eml_db[name]["sections"] = sorted(list(self.eml_db[name]["sections"]))
                self.eml_db[name]["indications"] = sorted(list(self.eml_db[name]["indications"]))
                self.eml_db[name]["atc_codes"] = sorted(list(self.eml_db[name]["atc_codes"]))
                
            logger.info(f"Successfully loaded {len(self.eml_db)} unique essential medicines from WHO EML.")
        except Exception as e:
            logger.error(f"Error loading WHO EML: {e}")
            self.eml_db = {}

    def get_metadata(self, generic_name: str) -> dict:
        """
        Looks up a generic drug name in the EML database.
        Returns a dictionary of EML metadata if found, else empty dict.
        """
        if not self.eml_db or not generic_name:
            return {}
        
        # Clean the query drug name
        cleaned_query = clean_medicine_name(generic_name)
        if not cleaned_query:
            return {}
            
        # Map common synonyms to match EML database keys
        synonyms = {
            "acetaminophen": "paracetamol",
            "acetylsalicylic acid": "aspirin",
            "acetylsalicylic": "aspirin"
        }
        if cleaned_query in synonyms:
            cleaned_query = synonyms[cleaned_query]
        
        # Try direct exact match
        if cleaned_query in self.eml_db:
            meta = self.eml_db[cleaned_query]
            return {
                "is_essential": True,
                "eml_section": ", ".join(meta["sections"]),
                "eml_indications": ", ".join(meta["indications"]),
                "atc_codes": ", ".join(meta["atc_codes"])
            }
        
        # Try partial matching (substring matching)
        # Often a drug name like "Metformin hydrochloride" should match "metformin"
        for eml_name, meta in self.eml_db.items():
            if eml_name in cleaned_query or cleaned_query in eml_name:
                return {
                    "is_essential": True,
                    "eml_section": ", ".join(meta["sections"]),
                    "eml_indications": ", ".join(meta["indications"]),
                    "atc_codes": ", ".join(meta["atc_codes"])
                }
                
        return {"is_essential": False}

    def get_alternatives(self, generic_name: str) -> List[Dict[str, Any]]:
        """
        Finds other essential medicines listed under the same EML sections.
        """
        cleaned_query = clean_medicine_name(generic_name)
        if not cleaned_query:
            return []
            
        # Map common synonyms to match EML database keys
        synonyms = {
            "acetaminophen": "paracetamol",
            "acetylsalicylic acid": "aspirin",
            "acetylsalicylic": "aspirin"
        }
        if cleaned_query in synonyms:
            cleaned_query = synonyms[cleaned_query]
            
        # Try direct exact match first, then fallback to partial
        found_key = None
        if cleaned_query in self.eml_db:
            found_key = cleaned_query
        else:
            for eml_name in self.eml_db:
                if eml_name in cleaned_query or cleaned_query in eml_name:
                    found_key = eml_name
                    break
                    
        if not found_key:
            return []
            
        query_sections = self.eml_db[found_key]["sections"]
        if not query_sections:
            return []
            
        alternatives = []
        for name, meta in self.eml_db.items():
            if name == found_key:
                continue
            # Check if there is intersection in sections
            common_sections = set(meta["sections"]).intersection(set(query_sections))
            if common_sections:
                alternatives.append({
                    "name": meta["original_name"],
                    "sections": meta["sections"],
                    "indications": meta["indications"],
                    "atc_codes": meta["atc_codes"]
                })
        return alternatives

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = WHOLoader()
    # Test lookups
    print("Metformin lookup:", loader.get_metadata("Metformin Hydrochloride"))
    print("Aspirin lookup:", loader.get_metadata("Acetylsalicylic acid (aspirin)"))
    print("Non-essential lookup:", loader.get_metadata("Sildenafil citrate"))
