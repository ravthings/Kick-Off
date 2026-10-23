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

with open('footbal.txt', 'r') as F:
    text = F.read
    print(text[:500])

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