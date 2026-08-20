import yaml
from dotenv import load_dotenv
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_config(path: str = "configs/config.yaml") -> dict:

    #Load .env file into os.environ first
    load_dotenv()

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded from {path}")
    return config