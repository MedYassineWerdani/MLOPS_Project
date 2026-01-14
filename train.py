from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import pandas as pd


checkpoint_folder = './checkpoints'


language_model = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(language_model)
device = "cpu"
# GPU can cause memory issues - using CPU for stability
if torch.cuda.is_available():
    device = "cuda"
print(f"{device=}")


# Load local TSV (expects columns 'text' and 'label')
import os
df = pd.read_csv("data/data.tsv", sep='\t')
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("data.tsv must contain 'text' and 'label' columns")
# If labels are strings, convert to integer categories
if df['label'].dtype == object:
    df['label'] = df['label'].astype('category').cat.codes
class LocalDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self._df = df.reset_index(drop=True)
    def to_pandas(self):
        return self._df
    def __len__(self):
        return len(self._df)
    def __getitem__(self, idx):
        row = self._df.iloc[idx]
        return {'text': row['text'], 'label': int(row['label'])}
dataset = {'train': LocalDataset(df)}




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


model = TransformerClassifier(language_model, 2).to(device)
print(f"{model=}")




batch_size = 32
n = len(dataset['train'])
train_data, validation_data, test_data= torch.utils.data.random_split(dataset['train'], [int(n * 0.7), int(n * 0.15), int(n * 0.15)])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

for i in train_loader:
    print(i['text'][0])
    break



loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def tokenize(text, device):
    tokens = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True
    )
    return tokens['input_ids'].to(device), tokens['attention_mask'].to(device)


def start_train(start_epoch , max_epochs , save_snapshots = True):

    start_epoch = 0
    max_epochs = 2
    save_snapshots = True

    if start_epoch != 0:
        model.load_state_dict(torch.load(f"{checkpoint_folder}/epoch-{start_epoch}.pth"))


    best_acc = 0

    for t in range(start_epoch+1, max_epochs+1):
        print(f"epoch {t}: ", end='')


        # TRAIN
        model.train()
        for row in tqdm(train_loader):
            tokens, attention_mask = tokenize(row['text'], device)
            label = row["label"].to(device)

            loss_fn(model(tokens, attention_mask), label).backward()
            optimizer.step()
            optimizer.zero_grad()

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
        print("error:", avg_error)
        print("accuracy:", accuracy)


        if save_snapshots and accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(),  f"{checkpoint_folder}/epoch-{t}.pth")

    print("BEST ACC:", best_acc)

        