from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import logging
import os

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
