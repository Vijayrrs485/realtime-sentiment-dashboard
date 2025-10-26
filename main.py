from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from models.sentiment_model import SentimentAnalyzer
import config

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="Real-time sentiment analysis API using BERT"
)

# Allow all origins for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = SentimentAnalyzer()

class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

@app.on_event("startup")
async def load_model():
    analyzer.load_model()

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running"}

@app.get("/health")
def health():
    return {"status": "OK", "model_loaded": analyzer.is_loaded()}

@app.post("/analyze")
def analyze_text( TextInput):
    try:
        return analyzer.analyze(data.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-analyze")
def batch_analyze( BatchInput):
    try:
        results = [analyzer.analyze(t) for t in data.texts]
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))