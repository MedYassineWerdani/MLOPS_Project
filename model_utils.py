import mlflow
import torch
import torch.nn as nn
import pandas as pd
import re
import nltk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords
from tqdm import tqdm

# Constants
LANGUAGE_MODEL = "FacebookAI/roberta-base"

class TransformerClassifier(nn.Module):
    def __init__(self, n_classes):
        super(TransformerClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(LANGUAGE_MODEL)
        layer_size = self.transformer.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(layer_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes)
        )

    def forward(self, x, attention_mask):
        with torch.no_grad():
            x = self.transformer(input_ids=x, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0, :]
        return self.classifier(x)

class PandasTextDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': torch.tensor(self.labels[idx], dtype=torch.long)}

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())


def train_model(data_path, epochs=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
    
    # Load and Preprocess
    df = pd.read_csv(data_path, sep='\t', usecols=['page_text', 'bias']).dropna()
    df['text'] = df['page_text'].apply(preprocess_text)
    
    # Map Labels
    label_map = {'left': 0, 'leaning-left': 0, 'center': 1, 'leaning-right': 2, 'right': 2}
    df['label'] = df['bias'].str.lower().map(label_map)
    df = df.dropna(subset=['label'])

    dataset = PandasTextDataset(df)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = TransformerClassifier(n_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    # --- MLFLOW LOGGING STARTS HERE ---
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("model_type", LANGUAGE_MODEL)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            tokens = tokenizer(batch['text'], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(tokens['input_ids'], tokens['attention_mask'])
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Log average loss per epoch to MLflow
        avg_loss = running_loss / len(train_loader)
        mlflow.log_metric("loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1} Loss: {avg_loss}")

    return model
