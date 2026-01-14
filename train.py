from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import re
import string
from nltk.corpus import stopwords
import nltk
import os



checkpoint_folder = './checkpoints'

# ensure checkpoints directory exists
os.makedirs(checkpoint_folder, exist_ok=True)

language_model = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(language_model)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
device



data = pd.read_csv('data/data.tsv', sep='\t')


# Download stopwords
nltk.download('stopwords', quiet=True)
# Use a lighter stopword set (remove stopwords but keep potentially meaningful ones)
all_stopwords = set(stopwords.words('english'))
# Keep words that might indicate bias
keep_words = {'government', 'business', 'country', 'people', 'president', 'trump', 'biden', 'democrat', 'republican', 'left', 'right', 'liberal', 'conservative'}
stop_words = all_stopwords - keep_words

# Load data
df = pd.read_csv('data/data.tsv', usecols=['page_text', 'bias'] , sep='\t')
df = df.dropna(subset=['page_text', 'bias']).reset_index(drop=True)
print(f"Loaded {len(df)} rows")
print(f"Label distribution:\n{df['bias'].value_counts()}")

# TEXT PREPROCESSING
def preprocess_text(text):
    """Comprehensive text preprocessing pipeline"""
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags and entities
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-z]+;', '', text)
    
    # Remove special characters and digits (keep letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove short words (< 3 chars) - but less aggressively
    words = text.split()
    words = [w for w in words if len(w) >= 2]  # Changed from >= 3 to >= 2
    
    # Remove stopwords (lighter set)
    words = [w for w in words if w not in stop_words]
    
    text = ' '.join(words)
    
    return text

# Apply preprocessing
print("\nPreprocessing text...")
df['text'] = df['page_text'].apply(preprocess_text)

# Remove rows with empty text after preprocessing
df = df[df['text'].str.len() > 0].reset_index(drop=True)
print(f"After preprocessing: {len(df)} rows")

# Encode labels
df['bias'] = pd.Categorical(df['bias'])
df['label'] = df['bias'].cat.codes
label_mapping_5 = dict(enumerate(df['bias'].cat.categories))
print(f"\nOriginal 5-class mapping: {label_mapping_5}")

# Consolidate to 3 classes: Left, Center, Right
def map_to_3_classes(label_code):
    """Map 5-class labels to 3-class labels"""
    original_label = label_mapping_5[label_code].lower()
    if 'left' in original_label:
        return 0  # Left
    elif 'right' in original_label:
        return 2  # Right
    else:  # center, least, etc.
        return 1  # Center

df['label'] = df['label'].map(map_to_3_classes)
label_mapping = {0: 'Left', 1: 'Center', 2: 'Right'}
print(f"\nConsolidated 3-class mapping: {label_mapping}")
print(f"Number of classes: {df['label'].nunique()}")
print(f"Samples per class:\n{df['label'].value_counts().sort_index()}")


# Create dataset
class PandasTextDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = df['page_text'].tolist()
        self.labels = df['label'].tolist()
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': torch.tensor(self.labels[idx], dtype=torch.long)}

dataset = PandasTextDataset(df)

# Define model with 5 classes - added dropout for regularization
class TransformerClassifier(nn.Module):
    def __init__(self, model_name, n_classes):
        super(TransformerClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        layer_size = self.transformer.config.hidden_size

        self.classifer = nn.Sequential(
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
        x = self.classifer(x)
        return x

model = TransformerClassifier(language_model, 3).to(device)
# print(f"{model=}")


# Create data loaders
batch_size = 32
n = len(dataset)
train_len = int(n * 0.7)
val_len = int(n * 0.15)
test_len = n - train_len - val_len

train_data, validation_data, test_data = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print(f"Total dataset: {n} samples")
print(f"Train: {len(train_data)} samples ({len(train_loader)} batches)")
print(f"Validation: {len(validation_data)} samples ({len(validation_loader)} batches)")
print(f"Test: {len(test_data)} samples ({len(test_loader)} batches)")
for i in train_loader:
    print(f"Sample text length: {len(i['text'][0])} chars")
    break



loss_fn = nn.CrossEntropyLoss()
# Lower learning rate for better convergence on different models
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, eps=1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

def tokenize(text, device):
    tokens = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    return tokens['input_ids'].to(device), tokens['attention_mask'].to(device)




def train_data(start_epoch=0 , max_epochs=2 , save_snapshots=True):
    start_epoch = start_epoch
    max_epochs = max_epochs  # More epochs for new models to converge
    save_snapshots = True

    if start_epoch != 0:
        model.load_state_dict(torch.load(f"{checkpoint_folder}/epoch-{start_epoch}.pth"))

    best_acc = 0
    patience = 0
    patience_limit = 4

    for t in range(start_epoch+1, max_epochs+1):
        print(f"\nepoch {t}: ", end='')

        # TRAIN
        model.train()
        train_loss = 0
        for row in tqdm(train_loader):
            tokens, attention_mask = tokenize(row['text'], device)
            label = row["label"].to(device)

            optimizer.zero_grad()
            pred = model(tokens, attention_mask)
            loss = loss_fn(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        print(f"train_loss: {train_loss / len(train_loader):.4f}, ", end='')

        # VALIDATE
        model.eval()
        total_loss = 0
        correct = 0
        print(f"validation: ", end='')

        with torch.no_grad():
            for row in tqdm(validation_loader):
                tokens, attention_mask = tokenize(row['text'], device)
                label = row["label"].to(device)
                pred = model(tokens, attention_mask)
                correct += (pred.argmax(1) == label).type(torch.float).sum().item()
                total_loss += loss_fn(pred, label).item()

        avg_error = total_loss / len(validation_loader)
        accuracy = correct / len(validation_loader.dataset)
        print("error: {:.4f}, accuracy: {:.4f}".format(avg_error, accuracy))
        
        # Learning rate scheduling
        scheduler.step(accuracy)

        if save_snapshots and accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(),  f"{checkpoint_folder}/epoch-{t}.pth")
            print(f"  → Saved checkpoint! Best so far: {best_acc:.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"  → Early stopping triggered (no improvement for {patience_limit} epochs)")
                break

    print(f"\nBEST ACC: {best_acc:.4f}")

train_data()