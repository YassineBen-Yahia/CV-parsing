import spacy
from spacy.tokens import DocBin
import random
from pathlib import Path
from tqdm import tqdm
import json
import re

def filter_non_overlapping_spans(spans):
    """
    Given a list of spaCy Span objects, return only non-overlapping spans.
    Uses a set to track covered token indices for overlap detection.
    """
    filtered = []
    covered = set()
    for span in sorted(spans, key=lambda s: (s.start, -s.end)):
        span_range = set(range(span.start, span.end))
        if not span_range & covered:
            filtered.append(span)
            covered.update(span_range)
    return filtered



def create_spacy_files(json_path, train_path, dev_path, exclude_entities,split_skill_entities, train_ratio=0.8):
    nlp = spacy.blank("en")
    data = []
    json_data = json.loads(Path(json_path))

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item["content"]
            entities = item["annotation"]
            entity_spans = []
            for annotation in entities:
                # Safely extract label
                if isinstance(annotation["label"], list):
                    if not annotation["label"]:
                        continue  # skip if label list is empty
                    label = annotation["label"][0]
                else:
                    label = annotation["label"]
                if label in exclude_entities:
                    continue
                if split_skill_entities and label == "Skills":
                    # Split the skill entity into individual skills
                    for point in annotation["points"]:
                        if not isinstance(point, dict):
                            continue
                        start = int(point["start"])
                        end = int(point["end"])
                        skill_text = text[start:end]
                        # Use comma/semicolon to split skills
                        if "," in skill_text or ";" in skill_text:
                            normalized = skill_text.replace(";", ",")
                            normalized = normalize_skills(normalized)
                            for skill in [s.strip() for s in normalized.split(",") if s.strip()]:
                                skill_start = text.find(skill, start, end)
                                if skill_start != -1:
                                    skill_end = skill_start + len(skill)
                                    if 0 <= skill_start < skill_end <= len(text):
                                        entity_spans.append((skill_start, skill_end, label))
                        else:
                            if 0 <= start < end <= len(text):
                                entity_spans.append((start, end, label))
                else:
                    for point in annotation["points"]:
                        if not isinstance(point, dict):
                            continue
                        start = int(point["start"])
                        end = int(point["end"])
                        if start is not None and end is not None:
                            try:
                                start, end = int(start), int(end)
                                # Skip invalid ranges
                                if not (0 <= start < end <= len(text)):
                                    continue
                                entity_spans.append((start, end, label))
                            except (ValueError, TypeError):
                                continue
            data.append((text, {"entities": entity_spans}))
    random.shuffle(data)
    #print(data[:2])  # Debug: print first 2 samples to verify content
    split = int(len(data) * train_ratio)
    data = [(text, {"entities": clean_entities(text, entities["entities"])}) for text, entities in data]    
    train_data = data[:split]
    #print(f"First training sample: {train_data[0]}")  # Debug: print first training sample
    dev_data = data[split:]
    for dataset, out_path in [(train_data, train_path), (dev_data, dev_path)]:
        doc_bin = DocBin()
        for text, entities in tqdm(dataset, desc=f"Processing {out_path}"):
            doc = nlp.make_doc(text)
            spans = []
            for start, end, label in entities["entities"]:
                span = doc.char_span(start, end, label=label, alignment_mode="strict")
                if span is not None:
                    spans.append(span)
            filtered_spans = filter_non_overlapping_spans(spans)
            doc.ents = filtered_spans
            doc_bin.add(doc)
        doc_bin.to_disk(out_path)
        print(f"Saved train to {train_path}, dev to {dev_path}")
    print("Saved !!!")

    return data[:split], data[split:]







def normalize_skills(text):
    """
    Normalize skill entries with comprehensive pattern recognition.
    """
    # Skip section headers and non-skill text
    if text.strip().upper() in ["SKILLS", "SKILLS:", "SKILL SETS", "COMPUTER LANGUAGES KNOWN:"]:
        return ""
    
    # Extract experience info in parentheses (preserve it for output)
    exp_pattern = r'\s*\(([^)]*(?:year|month|yr)[^)]*)\)'
    exp_match = re.search(exp_pattern, text, re.I)
    
    if exp_match:
        experience_info = exp_match.group(1)
        text = re.sub(exp_pattern, '', text)
    
    # Remove formatting elements and clean up text
    text = re.sub(r'^[•\-\*\d]+\.?\s*', '', text)
    text = re.sub(r'^\s*[–•:]\s*', '', text)
    text = re.sub(r'[.;:]$', '', text.strip())
    
    # Skip non-skill entries
    if text.lower().strip() in ["teaching", "polysaccarides'"]:
        return ""
    
    # Technology capitalization map
    skill_mapping = {
        'javascript': 'JavaScript',
        'java': 'Java',
        'python': 'Python',
        'c++': 'C++',
        'angular js': 'AngularJS',
        'html': 'HTML',
        'css': 'CSS',
        'aws': 'AWS',
        'sql': 'SQL',
        'docker': 'Docker',
        'git': 'Git',
        'ms office': 'Microsoft Office',
        'microsoft office': 'Microsoft Office',
        'excel': 'Excel',
        'end user computing': 'End User Computing',
        'active directory': 'Active Directory',
        'tally': 'Tally',
        'velocity': 'Velocity'
    }
    
    # Apply skill-specific capitalization
    for skill, proper_form in skill_mapping.items():
        text = re.sub(r'\b' + re.escape(skill) + r'\b', proper_form, text, flags=re.I)
    
    # Reattach experience info if present
    if exp_match:
        text = f"{text} ({experience_info})"
    
    return text.strip()

def clean_entities(text, entities):
    cleaned = []
    for ent in entities:
        
        start, end, label = ent
        new_start = start
        new_end = end
        # Remove leading whitespace
        while new_start < new_end and text[new_start].isspace():
            new_start += 1
        # Remove trailing whitespace
        while new_end > new_start and text[new_end - 1].isspace():
            new_end -= 1
        # Only add if non-empty after trimming
        if new_start < new_end:
            cleaned.append((new_start, new_end, label))
    return cleaned




def create_spacy_files2(json_path,exclude_entities,train_path,dev_path, train_ratio=0.8):
    nlp = spacy.blank("en")
    data = []
    json_data = json.load(open(Path(json_path)))
    print(json_data[0][1]['entities'])
    for cv in json_data:
            
        text = cv[0]
        entities = cv[1]['entities']
        entity_spans = []
        for annotation in entities:
            # Safely extract label
            label = annotation[2]
            if label in exclude_entities:
                continue

            point={"start":annotation[0],"end":annotation[1]}
            
            
            if not isinstance(point, dict):
                continue
            start = int(point["start"])
            end = int(point["end"])
            if start is not None and end is not None:
                try:
                    start, end = int(start), int(end)
                    # Skip invalid ranges
                    if not (0 <= start < end <= len(text)):
                        continue
                    entity_spans.append((start, end, label))
                except (ValueError, TypeError):
                    continue
        data.append((text, {"entities": entity_spans}))
    random.shuffle(data)
    #print(data[:2])  # Debug: print first 2 samples to verify content
    split = int(len(data) * train_ratio)
    data = [(text, {"entities": clean_entities(text, entities["entities"])}) for text, entities in data]    
    train_data = data[:split]
    #print(f"First training sample: {train_data[0]}")  # Debug: print first training sample
    dev_data = data[split:]
    for dataset, out_path in [(train_data, train_path), (dev_data, dev_path)]:
        doc_bin = DocBin()
        for text, entities in tqdm(dataset, desc=f"Processing {out_path}"):
            doc = nlp.make_doc(text)
            spans = []
            for start, end, label in entities["entities"]:
                span = doc.char_span(start, end, label=label, alignment_mode="strict")
                if span is not None:
                    spans.append(span)
            filtered_spans = filter_non_overlapping_spans(spans)
            doc.ents = filtered_spans
            doc_bin.add(doc)
        doc_bin.to_disk(out_path)
        print(f"Saved train to {train_path}, dev to {dev_path}")
    print("Saved !!!")

    return data[:split], data[split:]






if __name__ == "__main__":
    #path=Path(r"C:\ML\CV-Parsing\Data\training\train_data.json")
    #create_spacy_files2(path)

    train_data, val_data = create_spacy_files2(
        json_path=r"C:\ML\CV-Parsing\Data\training\train_data.json",
        train_path=r"C:\ML\CV-Parsing\Data\train.spacy",
        dev_path=r"C:\ML\CV-Parsing\Data\dev.spacy",
        exclude_entities=["UNKNOWN","Graduation Year","Years of Experience"],
        train_ratio=0.8,
    )