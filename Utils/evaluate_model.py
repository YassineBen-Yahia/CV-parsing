import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
model_path = Path(r"C:\ML\CV-Parsing\training_output\model-best")  #  trained model
dev_data_path = Path(r"C:\ML\CV-Parsing\Data\dev.spacy")           #  dev set
metrics_output = Path(r"C:\ML\CV-Parsing\metrics.json")            #  output file

# Run spaCy evaluation
logger.info("Evaluating spaCy model...")

result = subprocess.run(
    [
        "python", "-m", "spacy", "evaluate",
        str(model_path),
        str(dev_data_path),
        "--output", str(metrics_output)
    ],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    logger.info("Evaluation completed successfully!")
    logger.info(result.stdout)   
else:
    logger.error("Evaluation failed!")
    logger.error(result.stderr)
logger.info("Evaluation process finished.")