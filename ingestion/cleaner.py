import re
import logging

logger = logging.getLogger(__name__)

def clean_html(text: str) -> str:
    """
    Removes HTML tags and clean double spaces.
    """
    if not text:
        return ""
    # Strip HTML tags
    text = re.sub(r'<[^>]*>', ' ', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_drug_field(field_value) -> str:
    """
    Cleans drug fields that can be lists of strings or single strings.
    Converts list to a single concatenated string.
    """
    if field_value is None:
        return ""
    
    if isinstance(field_value, list):
        # Filter out empty entries and join with newline
        cleaned_items = [clean_html(str(item)) for item in field_value if item]
        return "\n".join(cleaned_items)
    
    return clean_html(str(field_value))

def extract_names(openfda_dict: dict) -> tuple:
    """
    Extracts brand_name and generic_name from openfda dictionary.
    Returns (brand_name, generic_name) as strings.
    """
    if not openfda_dict:
        return "", ""
    
    brand_list = openfda_dict.get("brand_name", [])
    generic_list = openfda_dict.get("generic_name", [])
    
    brand_name = brand_list[0] if brand_list else ""
    generic_name = generic_list[0] if generic_list else ""
    
    # Clean them up
    brand_name = clean_html(str(brand_name))
    generic_name = clean_html(str(generic_name))
    
    return brand_name, generic_name
