import subprocess
import spacy
import random
import logging
from pathlib import Path
from spacy.tokens import DocBin
from tqdm import tqdm
import json
from tools import create_spacy_files


def load_and_export_ner_data(
    train_ratio=0.8,
    run_validation=False,
    exclude_entities=["UNKNOWN","Graduation Year","Years of Experience"],
    split_skill_entities=True,
):
    """
    Enhanced resume NER data loader with advanced cleaning and analysis.
    """
    
    #ner_dataset_path = Path('/kaggle/input/resume-entities-for-ner/Entity Recognition in Resumes.json')
    ner_dataset_path = Path(r'C:\ML\CV-Parsing\Data\Entity Recognition in Resumes.json')

    train_data , dev_data = create_spacy_files(
        json_path=r"C:\ML\CV-Parsing\Data\Entity Recognition in Resumes.json",
        train_path=r"C:\ML\CV-Parsing\Data\train.spacy",
        dev_path=r"C:\ML\CV-Parsing\Data\dev.spacy",
        exclude_entities=exclude_entities,
        split_skill_entities=split_skill_entities,
        train_ratio=0.8
    )
    return{
        'train_data': train_data, 
        'val_data': dev_data,
    }


if __name__ == "__main__":
    load_and_export_ner_data()
    
