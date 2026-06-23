import uuid
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class NativeRecursiveTextSplitter:
    """
    A pure Python implementation of RecursiveCharacterTextSplitter.
    Avoids external dependencies like langchain.
    """
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        # Start recursion
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        # If text is already small enough, return it
        if len(text) <= self.chunk_size:
            return [text]
            
        # If no separators left, split by raw length
        if not separators:
            chunks = []
            start = 0
            while start < len(text):
                chunks.append(text[start:start + self.chunk_size])
                start += self.chunk_size - self.chunk_overlap
                if start >= len(text) or self.chunk_size <= self.chunk_overlap:
                    break
            return chunks

        # Find the first separator that actually splits the text
        separator = separators[0]
        next_separators = separators[1:]
        
        # Split text by separator
        if separator == "":
            # Split into individual characters
            splits = list(text)
        else:
            # We want to keep the separator if possible, or just split on it
            # To keep it simple, we split on it
            splits = text.split(separator)
            
        # Reconstruct splits: we merge them back up to chunk_size
        chunks = []
        current_doc = []
        current_len = 0
        
        for split in splits:
            # If the split itself is too large, recursively split it
            if len(split) > self.chunk_size:
                # Flush current buffer first
                if current_doc:
                    chunks.append(separator.join(current_doc))
                    current_doc = []
                    current_len = 0
                
                # Recursively split the large piece
                sub_splits = self._split_text(split, next_separators)
                chunks.extend(sub_splits)
            else:
                # Calculate size after adding this split
                # If current_doc is not empty, we add the separator length
                sep_len = len(separator) if current_doc else 0
                if current_len + sep_len + len(split) > self.chunk_size:
                    # Flush current buffer
                    if current_doc:
                        chunks.append(separator.join(current_doc))
                    
                    # Create overlap buffer: search backwards in current_doc to form overlap
                    # For simplicity, we can do a basic overlap by taking the last split(s)
                    # or taking characters. Let's build overlap from current_doc
                    overlap_doc = []
                    overlap_len = 0
                    for item in reversed(current_doc):
                        item_sep_len = len(separator) if overlap_doc else 0
                        if overlap_len + item_sep_len + len(item) <= self.chunk_overlap:
                            overlap_doc.insert(0, item)
                            overlap_len += item_sep_len + len(item)
                        else:
                            break
                    
                    current_doc = overlap_doc
                    current_len = overlap_len
                
                current_doc.append(split)
                current_len += (len(separator) if len(current_doc) > 1 else 0) + len(split)
                
        if current_doc:
            chunks.append(separator.join(current_doc))
            
        return chunks

class MedicalChunkSplitter:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = NativeRecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def split_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits a cleaned drug record into sections and chunks each section separately.
        This preserves the boundary and context of each label section.
        """
        chunks = []
        drug_id = record.get("id")
        brand_name = record.get("brand_name", "")
        generic_name = record.get("generic_name", "")
        eml_meta = record.get("eml_metadata", {})
        
        # Sections to split
        sections = [
            "indications_and_usage",
            "warnings",
            "contraindications",
            "dosage_and_administration",
            "adverse_reactions",
            "drug_interactions"
        ]
        
        for section in sections:
            section_text = record.get(section, "")
            if not section_text or not section_text.strip():
                continue
                
            # Split the section text
            text_splits = self.splitter.split_text(section_text)
            
            for idx, split_text in enumerate(text_splits):
                chunk_id = f"{drug_id}_{section}_{idx}"
                
                # Combine metadata fields
                metadata = {
                    "drug_id": drug_id,
                    "brand_name": brand_name,
                    "generic_name": generic_name,
                    "section": section,
                    "chunk_id": chunk_id,
                    "is_essential": eml_meta.get("is_essential", False),
                    "eml_section": eml_meta.get("eml_section", "N/A"),
                    "eml_indications": eml_meta.get("eml_indications", "N/A"),
                    "atc_codes": eml_meta.get("atc_codes", "N/A")
                }
                
                # The text that will be embedded
                # We prefix it with the drug names and section to enrich retrieval semantics
                enriched_text = f"Drug: {brand_name} ({generic_name})\nSection: {section.replace('_', ' ').title()}\nContent: {split_text}"
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": enriched_text,
                    "raw_content": split_text,
                    "metadata": metadata
                })
                
        return chunks

if __name__ == "__main__":
    splitter = MedicalChunkSplitter()
    test_rec = {
        "id": "123-abc",
        "brand_name": "Tylenol",
        "generic_name": "Acetaminophen",
        "indications_and_usage": "For temporary relief of minor aches and pains due to headache, muscular aches, minor pain of arthritis. Also helpful for reduce fever.",
        "warnings": "Liver warning: This product contains acetaminophen. Severe liver damage may occur if you take more than 4,000 mg of acetaminophen in 24 hours.",
        "eml_metadata": {
            "is_essential": True,
            "eml_section": "Analgesics",
            "eml_indications": "Pain relief",
            "atc_codes": "N02BE01"
        }
    }
    chunks = splitter.split_record(test_rec)
    print(f"Generated {len(chunks)} chunks.")
    for c in chunks:
        print(f"Chunk ID: {c['chunk_id']}")
        print(f"Text:\n{c['text']}\n")
