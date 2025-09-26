# train_tcn.py
# pip install torch numpy scikit-learn
import os, csv, math, argparse, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

NUM_CLASSES = 5  # [neg, help, open, fist, wave]

class SeqDS(Dataset):
    def __init__(self, rows, seq_len=48):
        self.rows = rows
        self.seq_len = seq_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        path, lab = self.rows[i]
        d = np.load(path)
        X = d["seq"].astype(np.float32)  # [T, 128]
        y = int(d["label"])
        T = X.shape[0]
        if T < self.seq_len:
            pad = np.repeat(X[-1:], self.seq_len-T, axis=0)
            X = np.concatenate([X, pad], axis=0)
        elif T > self.seq_len:
            X = X[-self.seq_len:]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)

class TinyTCN(nn.Module):
    def __init__(self, in_dim=128, n_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 96, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 96, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Linear(96, n_classes)
    def forward(self, x):          # x: [B,T,F]
        x = x.transpose(1,2)       # -> [B,F,T]
        x = self.net(x).squeeze(-1)# -> [B,96]
        return self.head(x)

def load_index(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append((r["path"], int(r["label"])))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="model_tcn.pth")
    args = ap.parse_args()

    rows = load_index(args.index)
    if not rows:
        raise RuntimeError("No data rows found in index.csv")

    # simple random split; for rigor, switch to subject-wise split if you encode subject IDs in filenames
    tr, va = train_test_split(rows, test_size=0.2, random_state=42, stratify=[r[1] for r in rows])
    tr_ds = SeqDS(tr, seq_len=args.seq_len)
    va_ds = SeqDS(va, seq_len=args.seq_len)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTCN().to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for ep in range(1, args.epochs+1):
        # train
        model.train(); tot=0; correct=0
        for X,y in tr_dl:
            X,y = X.to(device), y.to(device)
            logits = model(X)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1)==y).sum().item()
            tot += y.size(0)
        tr_acc = correct/tot

        # validate
        model.eval(); tp=[0]*NUM_CLASSES; fp=[0]*NUM_CLASSES; fn=[0]*NUM_CLASSES
        with torch.no_grad():
            for X,y in va_dl:
                X,y = X.to(device), y.to(device)
                pred = model(X).argmax(1)
                for c in range(NUM_CLASSES):
                    tp[c] += ((pred==c)&(y==c)).sum().item()
                    fp[c] += ((pred==c)&(y!=c)).sum().item()
                    fn[c] += ((pred!=c)&(y==c)).sum().item()
        # macro F1
        f1s=[]
        for c in range(NUM_CLASSES):
            precision = tp[c]/(tp[c]+fp[c]+1e-9)
            recall    = tp[c]/(tp[c]+fn[c]+1e-9)
            f1s.append(2*precision*recall/(precision+recall+1e-9))
        f1 = float(np.mean(f1s))

        print(f"Epoch {ep:02d} | train_acc {tr_acc:.3f} | val_macroF1 {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.out)
            print("  saved:", args.out)

if __name__ == "__main__":
    main()
