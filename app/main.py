from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import time

app = FastAPI(
    title="Text Paraphraser API",
    description="API for paraphrasing text using rut5-base-paraphraser model",
    version="1.0.0"
)

model_name = "cointegrated/rut5-base-paraphraser"
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        print(f"Loading model on {device}...")
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries} to load model...")
                tokenizer = T5Tokenizer.from_pretrained(
                    model_name,
                    legacy=False,
                    local_files_only=False
                )
                model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    local_files_only=False
                )
                model.to(device)
                model.eval()
                print("Model loaded successfully!")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error loading model (attempt {attempt + 1}): {str(e)}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to load model after {max_retries} attempts: {str(e)}")
                    raise


@app.on_event("startup")
async def startup_event():
    load_model()


class ParaphraseRequest(BaseModel):
    text: str
    num_return_sequences: int = 1
    num_beams: int = 5
    temperature: float = 1.0


class ParaphraseResponse(BaseModel):
    original_text: str
    paraphrases: list[str]
    device: str


@app.post("/paraphrase", response_model=ParaphraseResponse)
async def paraphrase(request: ParaphraseRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_ids = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                num_return_sequences=request.num_return_sequences,
                num_beams=request.num_beams,
                temperature=request.temperature,
                max_length=256,
                do_sample=True
            )
        
        paraphrases = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return ParaphraseResponse(
            original_text=request.text,
            paraphrases=paraphrases,
            device=device
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during paraphrasing: {str(e)}")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "cuda_available": torch.cuda.is_available()
    }
