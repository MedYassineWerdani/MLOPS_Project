from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import os

# Initialize FastAPI
app = FastAPI(title="Fake News Detection API", version="1.0.0")

# Device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# Load model components
language_model = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(language_model)
checkpoint_folder = './checkpoints'

# Label mapping (0 = Real, 1 = Fake)
label_mapping = {
    0: 'Real News',
    1: 'Fake News'
}

# Model Definition
class TransformerClassifier(nn.Module):
    def __init__(self, model_name, n_classes):
        super(TransformerClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        layer_size = self.transformer.config.hidden_size

        self.classifer = nn.Sequential(
            nn.Linear(layer_size, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x, attention_mask):
        with torch.no_grad():
            x = self.transformer(input_ids=x, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.classifer(x)
        return x

# Load model
try:
    model = TransformerClassifier(language_model, 2).to(device)
    # Try to load the best checkpoint
    checkpoint_path = f"{checkpoint_folder}/epoch-4.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using untrained model.")
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")

# Tokenization function
def tokenize(text, device):
    tokens = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    return tokens['input_ids'].to(device), tokens['attention_mask'].to(device)

# Request/Response Models
class TextInput(BaseModel):
    text: str

class FakeNewsResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    is_fake: bool
    probabilities: dict

# Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "device": device, "model": "Fake News Detection"}

@app.post("/predict", response_model=FakeNewsResponse)
async def predict_fake_news(input_data: TextInput):
    """Predict if text is fake news or real news"""
    try:
        if not input_data.text or len(input_data.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Tokenize
        tokens, attention_mask = tokenize(input_data.text, device)
        
        # Predict
        with torch.no_grad():
            logits = model(tokens, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(1).item()
            confidence = probabilities[0, pred_class].item()
        
        # Get probabilities
        probs = {
            label_mapping[i]: float(probabilities[0, i].item())
            for i in range(len(label_mapping))
        }
        
        return FakeNewsResponse(
            text=input_data.text[:200] + "..." if len(input_data.text) > 200 else input_data.text,
            prediction=label_mapping[pred_class],
            confidence=confidence,
            is_fake=(pred_class == 1),
            probabilities=probs
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(texts: list[str]):
    """Predict fake news for multiple texts"""
    try:
        results = []
        for text in texts:
            if not text or len(text.strip()) == 0:
                results.append({"error": "Empty text"})
                continue
            
            tokens, attention_mask = tokenize(text, device)
            
            with torch.no_grad():
                logits = model(tokens, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                pred_class = logits.argmax(1).item()
                confidence = probabilities[0, pred_class].item()
            
            probs = {
                label_mapping[i]: float(probabilities[0, i].item())
                for i in range(len(label_mapping))
            }
            
            results.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "prediction": label_mapping[pred_class],
                "confidence": confidence,
                "is_fake": (pred_class == 1),
                "probabilities": probs
            })
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
