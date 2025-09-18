# CV-Parsing Project

This project extracts and analyzes entities from resumes using spaCy, transformer models, and PDF parsing. It includes data augmentation, visualization, and a chatbot interface.

## Features
- Named Entity Recognition (NER) for resumes
- Transformer-based spaCy model training
- PDF resume parsing
- Data augmentation and cleaning
- Data visualization (entity distribution, most frequent entities)
- Chatbot integration (OpenAI API)

## Folder Structure
- `Data/` — Raw and processed data files (except train.spacy and dev.spacy)
- `Utils/` — Utility scripts for data loading, augmentation, visualization, etc.
- `train_with_tarnsformer.py` — Script for training spaCy transformer model
- `chatbot.py` — Chatbot interface using OpenAI
- `train models/try_model.ipynb` — Jupyter notebook for model inference and demo

## Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your data in the `Data/` folder
4. Train the model using `train_with_tarnsformer.ipynb`
5. Run the notebook or chatbot for inference

## Requirements
See `requirements.txt` for all dependencies.

## Usage
- Train the model:
  ```bash
  python train_with_tarnsformer.py
  ```
- Run the chatbot:
  ```bash
  python chatbot.py
  ```
    chatbot's answer : 
    1. **Data Visualization**: Learn tools like Tableau or Power BI to present data effectively.
    2. **Machine Learning**: Familiarize yourself with ML algorithms and libraries (e.g., Scikit-learn, TensorFlow).
    3. **Statistical Analysis**: Enhance your understanding of statistics and analytical techniques for deeper insights.



- Visualize data:
  ```bash
  python Utils/visualization.py
  ```
- Try the model in Jupyter:
  Open `train models/try_model.ipynb`

## Datasets
- [Link text](https://www.kaggle.com/datasets/atharvasankhe/resumeparsing)
- [Link text](https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner)

