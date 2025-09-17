"""import subprocess
import logging
from pathlib import Path
import spacy

import torch


train_output_path = Path(r"C:\ML\CV-Parsing\Data\augmented_training_data.spacy")  
dev_output_path = Path(r"C:\ML\CV-Parsing\Data\dev.spacy")   

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


base_config_path = Path(r"Data\training\base_config.cfg")

logger.info("Initializing spaCy config...")
config_path = Path("./config.cfg")
if not config_path.exists():
    subprocess.run(
        ["python", "-m", "spacy", "init", "fill-config", base_config_path, "config.cfg"],
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

output_dir = Path(r"C:\ML\CV-Parsing\transformer")  # where the trained model will be saved
output_dir.mkdir(parents=True, exist_ok=True)

# Check for GPU availability
gpu_args = []
try:
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}. Training will use GPU.")
        gpu_args = ["--gpu-id", "0"]
    else:
        logger.info("No GPU detected. Training will use CPU.")
except Exception as e:
    logger.warning(f"Could not check GPU status: {e}")

train_cmd = [
    "python", "-m", "spacy", "train", "config.cfg",
    "--output", str(output_dir),
    "--paths.train", str(train_output_path),
    "--paths.dev", str(dev_output_path)
] + gpu_args

train_result = subprocess.run(
    train_cmd,
    capture_output=True, text=True
)

if train_result.returncode == 0:
    logger.info("Training completed successfully!")
    logger.info(train_result.stdout)
else:
    logger.error("Training failed!")
    logger.error(train_result.stderr)
logger.info("Training process finished.")"""

import torch

print (torch.cuda.is_available())
print (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

print(torch.__version__)       # Torch version
print(torch.version.cuda)      # CUDA version it was built with
print(torch.cuda.is_available()) 
print(torch.cuda.device_count())
