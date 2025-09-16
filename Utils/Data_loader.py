import subprocess
import spacy
import random
import logging
from pathlib import Path
from spacy.tokens import DocBin
from tqdm import tqdm
import json


def load_and_export_ner_data(
    train_ratio=0.8,
    train_output_path="./train.spacy", 
    dev_output_path="./dev.spacy",
    debug_mode=False,
    run_validation=False,
    visualize_patterns=False, 
    exclude_entities=["UNKNOWN","Graduation Year","Years of Experience"],
    analyze_before_processing=False,
    split_skill_entities=True,
):
    """
    Enhanced resume NER data loader with advanced cleaning and analysis.
    """
    
    #ner_dataset_path = Path('/kaggle/input/resume-entities-for-ner/Entity Recognition in Resumes.json')
    ner_dataset_path = Path(r'C:\ML\CV-Parsing\Data\Entity Recognition in Resumes.json')

    if not ner_dataset_path.exists():
        logging.error(f"Dataset file not found at {ner_dataset_path}")
        return None, None
    
    nlp = spacy.blank("en")
    db = DocBin()

    raw_data = []
    all_data = []

    logging.info("Extracting raw entities for analysis...")
    try:
        with open(ner_dataset_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading JSON data"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    
                    if 'content' not in item or 'annotation' not in item:
                        continue
                    
                    text = item['content']
                    entities = []
                    
                    for annotation in item['annotation']:
                        if not isinstance(annotation, dict) or 'label' not in annotation or 'points' not in annotation:
                            continue
                        
                        label = annotation['label'][0] if isinstance(annotation['label'], list) else annotation['label']
                        
                        # Skip excluded entity types
                        if label in exclude_entities:
                            continue
                        
                        for point in annotation['points']:
                            if not isinstance(point, dict):
                                continue
                            
                            start = point.get('start')
                            end = point.get('end')
                            
                            if start is not None and end is not None:
                                try:
                                    start, end = int(start), int(end)
                                    
                                    # Skip invalid ranges
                                    if not (0 <= start < end <= len(text)):
                                        continue
                                    
                                    entities.append((start, end, label))
                                except (ValueError, TypeError):
                                    continue
                    
                    if entities:
                        raw_data.append((text, {"entities": entities}))
                
                except Exception as e:
                    continue
    
    except Exception as e:
        logging.error(f"Error loading raw data: {e}")
        return None, None
    
    print(raw_data[:2])




if __name__ == "__main__":
    load_and_export_ner_data()
    
