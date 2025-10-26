import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 512

API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Sentiment Analysis API"
API_VERSION = "1.0.0"