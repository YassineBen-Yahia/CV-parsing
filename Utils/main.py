# utils/main.py
# This script runs the NER data pipeline: load, augment, visualize, and export.
import logging
from pathlib import Path
from Data_loader import load_and_export_ner_data
from augmentation import augment_and_balance_data
from visualization import visualize_data   


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading and exporting NER data...")
    train_data, val_data = load_and_export_ner_data()

    logger.info("Augmenting and balancing training data...")
    augmented_train_data = augment_and_balance_data(train_data)

    logger.info("Visualizing data distributions...")
    visualize_data()

    logger.info("Pipeline completed successfully.")