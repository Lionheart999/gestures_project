# gesture_temporal.py
# unified temporal gesture pipeline (0..9) with 4 model types:
#   - tcn (default)
#   - lstm
#   - gru
#   - transformer
#
# 0 = no gesture
# 1 = open palm
# 2 = fist
# 3 = thumbs up
# 4 = pointing
# 5 = victory / peace
# 6 = ok sign
# 7 = wave
# 8 = help signal
# 9 = swipe

import os, time, uuid, argparse, csv
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =========================
# Global config
# =========================
mp_hands = mp.solutions.hands

NUM_JOINTS = 21
FEAT_DIM = 128
SEQ_LEN_DEFAULT = 48

NUM_CLASSES = 10
LABEL_MAP = {
    "0": 0,  # no gesture / background
    "1": 1,  # open palm
    "2": 2,  # fist
    "3": 3,  # thumbs up
    "4": 4,  # pointing
    "5": 5,  # peace
    "6": 6,  # ok
    "7": 7,  # wave
    "8": 8,  # help
    "9": 9,  # circle
}

PAD_ZERO = np.zeros((NUM_JOINTS, 3), dtype=np.float32)

import random
def set_global_seed(seed: int = 42):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[seed fixed at {seed}]")

# call it immediately so all random ops are seeded
set_global_seed(42)

# =========================
# Normalization utilities
# =========================
def norm_palm_frame(lm):
    L = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17

    O = L[WRIST]
    a = L[INDEX_MCP] - O
    b = L[PINKY_MCP] - O

    u = b - a
    un = np.linalg.norm(u)
    u = u / un if un > 1e-8 else np.array([1, 0, 0], np.float32)

    v = L[MIDDLE_MCP] - O
    v = v - (u @ v) * u
    vn = np.linalg.norm(v)
    v = v / vn if vn > 1e-8 else np.array([0, 1, 0], np.float32)

    w = np.cross(u, v)
    wn = np.linalg.norm(w)
    w = w / wn if wn > 1e-8 else np.array([0, 0, 1], np.float32)

    R = np.stack([u, v, w], axis=1)
    X = (L - O) @ R

    scale = np.linalg.norm(((L[MIDDLE_MCP] - O) @ R)[:2]) + 1e-6
    Xn = X / scale

    mcp_idx = [5, 9, 13, 17]
    if Xn[mcp_idx, 2].mean() < 0:
        Xn[:, 2] *= -1.0

    return Xn


def handpack(frame_hands) -> np.ndarray:
    hands_sorted = {"Left": None, "Right": None}
    for lm, handedness in frame_hands:
        label = handedness.classification[0].label
        hands_sorted[label] = lm

    feats = []
    mask = []
    for side in ["Left", "Right"]:
        if hands_sorted[side] is None:
            feats.append(PAD_ZERO)
            mask.append(0.0)
        else:
            Xn = norm_palm_frame(hands_sorted[side].landmark)
            feats.append(Xn)
            mask.append(1.0)

    F = np.concatenate(feats, axis=0).reshape(-1)
    F = np.concatenate([F, np.array(mask, dtype=np.float32)])
    return F  # (128,)


# =========================
# Models (4 types)
# =========================
class TinyTCN(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, n_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 96, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 96, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Linear(96, n_classes)

    def forward(self, x):        # [B, T, F]
        x = x.transpose(1, 2)    # -> [B, F, T]
        x = self.net(x).squeeze(-1)
        return self.head(x)


class TinyLSTM(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, hidden=96, n_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):        # [B, T, F]
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class TinyGRU(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, hidden=96, n_classes=NUM_CLASSES):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


class TinyTransformer(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, n_classes=NUM_CLASSES, n_heads=4, depth=2, d_model=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=256,
            dropout=0.1,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, x):        # [B, T, F]
        x = self.proj(x)
        x = self.enc(x)
        x = x.mean(dim=1)
        return self.cls(x)


def make_model(model_type: str, seq_len: int = SEQ_LEN_DEFAULT):
    model_type = model_type.lower()
    if model_type == "tcn":
        return TinyTCN()
    if model_type == "lstm":
        return TinyLSTM()
    if model_type == "gru":
        return TinyGRU()
    if model_type == "transformer":
        return TinyTransformer()
    raise ValueError(f"unknown model_type: {model_type}")


# =========================
# Dataset
# =========================
class SeqDS(Dataset):
    def __init__(self, rows, seq_len=SEQ_LEN_DEFAULT):
        self.rows = rows
        self.seq_len = seq_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        path, lab = self.rows[i]
        d = np.load(path)
        X = d["seq"].astype(np.float32)
        y = int(d["label"])
        T = X.shape[0]
        if T < self.seq_len:
            pad = np.repeat(X[-1:], self.seq_len - T, axis=0)
            X = np.concatenate([X, pad], axis=0)
        elif T > self.seq_len:
            X = X[-self.seq_len:]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)


def load_index(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append((r["path"], int(r["label"])))
    return rows


# =========================
# collect
# =========================
def cmd_collect(args):
    os.makedirs(args.outdir, exist_ok=True)
    index_path = os.path.join(args.outdir, "index.csv")
    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write("path,label\n")

    print("Collecting... keys: 0=none, 1=palm, 2=fist, 3=thumbs up, 4=pointing, 5=peace, 6=ok, 7=wave, 8=help, 9=circle, q=quit")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    buf: List[np.ndarray] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.flip:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        pairs = []
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                pairs.append((lm, hd))

        feat = handpack(pairs)
        buf.append(feat)
        if len(buf) > args.seq_len:
            buf.pop(0)

        cv2.putText(frame, f"buffer: {len(buf)}/{args.seq_len}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Collecting... keys: 0=none, 1=palm, 2=fist, 3=thumbs up, 4=pointing, 5=peace, 6=ok, 7=wave, 8=help, 9=circle, q=quit", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("collect_temporal", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break

        ch = chr(k) if 48 <= k <= 57 else None  # digits 0..9
        if ch in LABEL_MAP and len(buf) == args.seq_len:
            lbl = LABEL_MAP[ch]
            arr = np.stack(buf, axis=0)
            fname = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}-L{lbl}.npz"
            path = os.path.join(args.outdir, fname)
            np.savez_compressed(path, seq=arr, label=np.int64(lbl))
            with open(index_path, "a") as f:
                f.write(f"{path},{lbl}\n")
            print("saved:", path)

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


# =========================
# train
# =========================
def cmd_train(args):
    rows = load_index(args.index)
    if not rows:
        raise RuntimeError("No data rows found in index.csv")

    X_train, X_val = train_test_split(
        rows,
        test_size=0.2,
        random_state=42,
        stratify=[r[1] for r in rows]
    )

    tr_ds = SeqDS(X_train, seq_len=args.seq_len)
    va_ds = SeqDS(X_val, seq_len=args.seq_len)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.model_type, seq_len=args.seq_len).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0
        correct = 0
        for X, y in tr_dl:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == y).sum().item()
            tot += y.size(0)
        tr_acc = correct / tot

        # validate
        model.eval()
        tp = [0] * NUM_CLASSES
        fp = [0] * NUM_CLASSES
        fn = [0] * NUM_CLASSES
        with torch.no_grad():
            for X, y in va_dl:
                X, y = X.to(device), y.to(device)
                pred = model(X).argmax(1)
                for c in range(NUM_CLASSES):
                    tp[c] += ((pred == c) & (y == c)).sum().item()
                    fp[c] += ((pred == c) & (y != c)).sum().item()
                    fn[c] += ((pred != c) & (y == c)).sum().item()

        f1s = []
        for c in range(NUM_CLASSES):
            precision = tp[c] / (tp[c] + fp[c] + 1e-9)
            recall = tp[c] / (tp[c] + fn[c] + 1e-9)
            f1s.append(2 * precision * recall / (precision + recall + 1e-9))
        macro_f1 = float(np.mean(f1s))

        print(f"Epoch {ep:02d} | {args.model_type} | train_acc {tr_acc:.3f} | val_macroF1 {macro_f1:.3f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), args.out)
            print("  saved:", args.out)


# =========================
# infer
# =========================
def cmd_infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.model_type, seq_len=args.seq_len).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    window = deque(maxlen=args.seq_len)
    over_cnt = 0
    last_class = None
    last_fire = 0

    CLASSES = [str(i) for i in range(10)]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.flip:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        pairs = []
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                pairs.append((lm, hd))

        feat = handpack(pairs)
        window.append(feat)
        H, W = frame.shape[:2]

        if len(window) == args.seq_len:
            X = torch.from_numpy(np.stack(list(window))[None, :, :]).float().to(device)
            with torch.no_grad():
                prob = torch.softmax(model(X), dim=1)[0].cpu().numpy()

            y0 = 20
            for i, c in enumerate(CLASSES):
                col = (0, 255, 0) if c != "0" else (200, 200, 200)
                cv2.putText(frame, f"{c}: {prob[i]:.2f}", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                y0 += 24

            c_idx = int(prob.argmax())
            c_name = CLASSES[c_idx]

            if c_name != "0" and prob[c_idx] >= args.thresh:
                if last_class == c_idx:
                    over_cnt += 1
                else:
                    over_cnt = 1
                last_class = c_idx
            else:
                over_cnt = 0
                last_class = None

            if over_cnt >= args.hold:
                last_fire = time.time()
                over_cnt = 0
                cv2.putText(frame, f"TRIGGER: {c_name}", (10, H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if time.time() - last_fire < 1.0:
            cv2.rectangle(frame, (0, 0), (W, 50), (0, 0, 255), -1)
            cv2.putText(frame, "GESTURE DETECTED", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("infer_temporal", frame)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


# =========================
# main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Temporal gesture pipeline (0..9) with TCN/LSTM/GRU/Transformer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # collect
    c = sub.add_parser("collect", help="Collect sequences with webcam")
    c.add_argument("--outdir", default="data")
    c.add_argument("--seq_len", type=int, default=SEQ_LEN_DEFAULT)
    c.add_argument("--width", type=int, default=1280)
    c.add_argument("--height", type=int, default=720)
    c.add_argument("--flip", action="store_true")
    c.set_defaults(func=cmd_collect)

    # train
    t = sub.add_parser("train", help="Train model")
    t.add_argument("--index", default="data/index.csv")
    t.add_argument("--seq_len", type=int, default=SEQ_LEN_DEFAULT)
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--out", default="model_tcn.pth")
    t.add_argument("--model_type",
                   default="tcn",
                   choices=["tcn", "lstm", "gru", "transformer"])
    t.set_defaults(func=cmd_train)

    # infer
    i = sub.add_parser("infer", help="Live inference")
    i.add_argument("--model", default="model_tcn.pth")
    i.add_argument("--seq_len", type=int, default=SEQ_LEN_DEFAULT)
    i.add_argument("--width", type=int, default=1280)
    i.add_argument("--height", type=int, default=720)
    i.add_argument("--flip", action="store_true")
    i.add_argument("--thresh", type=float, default=0.7)
    i.add_argument("--hold", type=int, default=6)
    i.add_argument("--model_type",
                   default="tcn",
                   choices=["tcn", "lstm", "gru", "transformer"])
    i.set_defaults(func=cmd_infer)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
