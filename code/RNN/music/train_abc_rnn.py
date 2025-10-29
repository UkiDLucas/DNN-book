# train_abc_rnn.py
import math, os, random, sys, torch, torch.nn as nn
from pathlib import Path

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1) load corpus
paths = list(Path("data/abc").glob("**/*.abc"))
text_chunks = []
for p in paths:
    s = p.read_text(encoding="utf-8", errors="ignore")
    text_chunks.append("\n\n" + s.strip() + "\n\n")
text = "\n".join(text_chunks)
# tiny safety clean
text = text.replace("\r", "").replace("\t", " ")

# 2) build vocab
chars = sorted(set(text))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}
vocab_size = len(chars)

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join(itos[int(i)] for i in t)

data = encode(text)
n = int(0.95*len(data))
train, val = data[:n], data[n:]

# 3) dataloader
block_size = 256
batch_size = 64
def get_batch(split):
    d = train if split=="train" else val
    ix = torch.randint(len(d)-block_size-1, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# 4) model
class CharGRU(nn.Module):
    def __init__(self, vocab, emb=256, hidden=384, layers=2, p=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb)
        self.gru = nn.GRU(emb, hidden, num_layers=layers, batch_first=True, dropout=p)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(hidden, vocab)
    def forward(self, x, h=None):
        x = self.embed(x)
        y, h = self.gru(x, h)
        y = self.drop(y)
        logits = self.fc(y)
        return logits, h

model = CharGRU(vocab_size).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()

# 5) train
steps = 3000
eval_every = 250
model.train()
for step in range(1, steps+1):
    x,y = get_batch("train")
    logits,_ = model(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()

    if step % eval_every == 0:
        with torch.no_grad():
            xv,yv = get_batch("val")
            lv,_ = model(xv)
            vloss = loss_fn(lv.view(-1,vocab_size), yv.view(-1)).item()
        print(f"step {step}/{steps} train {loss.item():.3f} val {vloss:.3f}")

# 6) sampling
@torch.no_grad()
def sample(model, prompt, max_tokens=2000, temperature=0.9, top_k=20):
    model.eval()
    x = encode(prompt).unsqueeze(0).to(device)
    h = None
    out = [int(i) for i in x[0]]
    for _ in range(max_tokens):
        logits, h = model(x, h)
        logits = logits[:,-1,:] / max(1e-6, temperature)
        if top_k is not None:
            v, idx = torch.topk(logits, k=top_k)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, idx, v)
            logits = mask
        probs = torch.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        out.append(int(ix))
        x = ix
    return decode(torch.tensor(out))

seed_abc = (
"X:1\nT:New Tune\nM:6/8\nL:1/8\nK:Dmix\n|: d e f g a b | a g f e d d |"
)

gen = sample(model, seed_abc, max_tokens=1500, temperature=0.9, top_k=20)
Path("generated.abc").write_text(gen, encoding="utf-8")
print("Saved generated.abc")
