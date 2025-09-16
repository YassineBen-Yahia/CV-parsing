import subprocess
import logging
from pathlib import Path
import spacy


train_output_path = Path(r"C:\ML\CV-Parsing\Data\augmented_training_data.spacy")  
dev_output_path = Path(r"C:\ML\CV-Parsing\Data\dev.spacy")   

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


logger.info("Initializing spaCy config...")
config_path = Path("./config.cfg")
if not config_path.exists():
    subprocess.run(
        ["python", "-m", "spacy", "init", "config", "config.cfg", "--lang", "en", "--pipeline", "ner"],
        check=True
    )


logger.info("Validating spaCy data...")
logger.info(f"Training data path: {train_output_path}")

result = subprocess.run(
    ["python", "-m", "spacy", "debug", "data", 
        "config.cfg", 
        "--paths.train", train_output_path,
        "--paths.dev", dev_output_path],
    capture_output=True, text=True
)


logger.info("Starting spaCy training...")

output_dir = Path(r"C:\ML\CV-Parsing\training_output")  # where the trained model will be saved
output_dir.mkdir(parents=True, exist_ok=True)

train_result = subprocess.run(
    [
        "python", "-m", "spacy", "train", "config.cfg",
        "--output", str(output_dir),
        "--paths.train", str(train_output_path),
        "--paths.dev", str(dev_output_path)
    ],
    capture_output=True, text=True
)

if train_result.returncode == 0:
    logger.info("Training completed successfully!")
    logger.info(train_result.stdout)
else:
    logger.error("Training failed!")
    logger.error(train_result.stderr)
logger.info("Training process finished.")