import torch
import torch.nn as nn
import wikipediaapi
torch.__version__
device = "mps"

wiki = wikipediaapi.Wikipedia(
    user_agent="kickoff-football-bot/1.0 (aaravgarg1011@gmail.com)",  # Wikipedia requires this
    language="en",
)
clubs = ["FC Barcelona", "Manchester City F.C", "Liverpool F.C.", "Real Madrid CF"]
text = ""
for club in clubs:
    page = wiki.page(club)
    if page.exists():
        text += page.text + "\n\n"
        print(f"Got {club}: {len(page.text)} chars")
    else:
        print(f"NOT FOUND: {club}  (check the title)")
with open("football.txt", "w", encoding="utf-8") as f:
    f.write(text)

with open('football.txt', 'r') as F:
    text = F.read

words = text.split()

vocab = sorted(set(words))
stoi = {w: idx for idx, w in enumerate(vocab)}
itos = {idx: w for w, idx in stoi.items()}

def encode(s):
    return [stoi[w] for w in s.split]
def decode(ids):
    return " ".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#Hyperparameters
batch_size = 32
block_size = 64
vocab_size = len(vocab)

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i, i + block_size]for i in ix])
    y = torch.stack([d[i+1, i+block_size+1]for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
xb, yb = get_batch('train')

class KickOffModel(nn.Module):
    def __init__(self):
        super().__init()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)

    def forward(self, idx):
        logits = self.embedding(idx)
        return logits

model = KickOffModel().to(device)
logits = model(xb)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-2)
model.load_state_dict(torch.load("kickoff.pth"))

B, T, C = logits.shape
loss = loss_fn(logits.view(B*T + C), yb.view(B*T))
print(loss.item())

