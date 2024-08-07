from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import logging
<<<<<<< HEAD

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

# Load tokenizer and model
model_path = "shivamkumaramehta/Search-Shield"  # Replace with your model path
token = "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"  # Replace with your Hugging Face token

tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path, revision="main", token=token)

# Function to predict profanity
def predict_profanity(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    return predicted_class

# Endpoint to predict profanity
@app.post("/predict-profanity/")
async def predict_profanity_endpoint(input: TextInput, request: Request):
    try:
        logging.info(f"Received request: {await request.json()}")
        predicted_class = predict_profanity(input.text)
        logging.info(f"Predicted class: {predicted_class}")
        return {"predicted_class": predicted_class}
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Vercel doesn't need this part
# if __name__ == "__main__":
#     import uvicorn
#     logging.basicConfig(level=logging.INFO)
#     uvicorn.run(app, host="127.0.0.1", port=8001)
=======
import os
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

# Load tokenizer and model
model_path = "shivamkumaramehta/Search-Shield"  # Replace with your model path
token = "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"  # Replace with your Hugging Face token

tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path, revision="main", token=token)

# Function to predict profanity
def predict_profanity(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    return predicted_class

# Endpoint to predict profanity
@app.post("/predict-profanity/")
async def predict_profanity_endpoint(input: TextInput, request: Request):
    try:
        logging.info(f"Received request: {await request.json()}")
        predicted_class = predict_profanity(input.text)
        logging.info(f"Predicted class: {predicted_class}")
        return {"predicted_class": predicted_class}
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
>>>>>>> 1e444e8d86b551d6658ded2ca4d936a838156c5f
