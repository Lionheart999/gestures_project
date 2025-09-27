# gesture_singleframe.py
# pip install mediapipe opencv-python numpy torch scikit-learn

import os, csv, time, argparse, uuid, json
import numpy as np
import cv2
import mediapipe as mp
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import pandas as pd


# =========================
# Config
# =========================
CLASSES = ["not_help", "help"]   # binary as per report
LABEL_KEYS = {"n":0, "h":1}      # hotkeys during collection

mp_hands = mp.solutions.hands
NUM_JOINTS = 21
FEAT_DIM = 63  # 21 * (x,y,z)

torch.manual_seed(67)
np.random.seed(67)

torch.cuda.manual_seed(67)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# Landmark normalization (single hand)
# =========================
def landmarks_to_63(lm_list) -> Optional[np.ndarray]:
    """
    Convert MediaPipe landmarks to a 63-dim feature vector for ONE hand.
    Normalization:
      - translate so WRIST is origin
      - build palm basis (index-MCP, middle-MCP, pinky-MCP) to get rotation invariance
      - scale by wrist->middle-MCP in-plane length
      - flatten to 63 (21x3)
    Returns None if no landmarks provided.
    """
    if lm_list is None: return None
    L = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)  # [21,3]
    WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0,5,9,17

    O = L[WRIST]
    a = L[INDEX_MCP] - O
    b = L[PINKY_MCP] - O
    u = b - a; un = np.linalg.norm(u); u = u/un if un>1e-8 else np.array([1,0,0],np.float32)
    v = L[MIDDLE_MCP] - O
    v = v - (u@v)*u
    vn = np.linalg.norm(v); v = v/vn if vn>1e-8 else np.array([0,1,0],np.float32)
    w = np.cross(u,v); wn = np.linalg.norm(w); w = w/wn if wn>1e-8 else np.array([0,0,1],np.float32)

    R = np.stack([u,v,w], axis=1)     # world->palm
    X = (L - O) @ R                   # rotate & translate
    scale = np.linalg.norm(((L[MIDDLE_MCP]-O) @ R)[:2]) + 1e-6
    Xn = X / scale                    # normalize scale

    # make "into palm" negative w (flip if needed)
    mcp_idx = [5,9,13,17]
    if Xn[mcp_idx,2].mean() < 0:
        Xn[:,2] *= -1.0

    return Xn.reshape(-1)             # 63-d

# =========================
# Model: simple MLP
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, n_classes=2, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)  # logits

# =========================
# Data IO
# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_sample(outdir, feat63, label):
    ensure_dir(outdir)
    fname = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}-L{label}.npz"
    fpath = os.path.join(outdir, fname)
    np.savez_compressed(fpath, x=feat63.astype(np.float32), y=np.int64(label))
    return fpath

def load_rows(index_csv):
    rows=[]
    with open(index_csv, "r") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append((r["path"], int(r["label"])))
    return rows

def load_dataset(rows):
    X = []
    y = []
    for p, lab in rows:
        if not os.path.exists(p):
            print("WARNING: file missing, skipping:", p)
            continue
        d = np.load(p)
        X.append(d["x"].astype(np.float32))
        y.append(int(d["y"]))
    if len(X) == 0:
        raise RuntimeError("No valid samples found!")
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    return X, y


# =========================
# Subcommand: collect
# =========================
def cmd_collect(args):
    """
    Collect single-frame samples with hotkeys:
      'h' -> help(1)
      'n' -> not_help(0)
      'q'/ESC -> quit
    Saves one .npz per keypress, and regenerates index_single.csv at the end.
    """
    outdir = args.outdir
    ensure_dir(outdir)
    index_csv = os.path.join(outdir, "index_single.csv")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           model_complexity=1, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    print("Hotkeys: [h]=help (1), [n]=not_help (0), [q]=quit")

    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.flip: frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        feat63 = None
        if res.multi_hand_landmarks:
            feat63 = landmarks_to_63(res.multi_hand_landmarks[0].landmark)

        # HUD
        txt = "Hand: OK" if feat63 is not None else "Hand: NOT FOUND"
        cv2.putText(frame, txt, (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, "Press [h]=help, [n]=not_help, [q]=quit", (12,58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("collect_single", frame)

        k = cv2.waitKey(1) & 0xFF
        if k in [27, ord('q')]: break
        if k in [ord('h'), ord('n')] and feat63 is not None:
            lab = LABEL_KEYS[chr(k)]
            fpath = save_sample(outdir, feat63, lab)
            print("saved:", fpath)

    # Clean up
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

    # Regenerate CSV at the end
    npz_files = [f for f in os.listdir(outdir) if f.endswith(".npz")]
    with open(index_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path","label"])
        for npz in npz_files:
            data = np.load(os.path.join(outdir, npz))
            label = int(data["y"])
            writer.writerow([os.path.join(outdir, npz), label])
    print(f"Regenerated index CSV: {index_csv}, {len(npz_files)} samples")

# =========================
# Subcommand: train
# =========================
def cmd_train(args):

    rows = load_rows(args.index)
    if len(rows) < 20:
        print("WARNING: very few samples; consider collecting more.")

    # 70/15/15 split as per report
    X, y = load_dataset(rows)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=67, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=67, stratify=y_tmp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=FEAT_DIM, n_classes=2, p_drop=0.3).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)   # Adam optimizer
    crit = nn.CrossEntropyLoss()

    # early stopping
    best_val = 1e9; best_state=None; patience=args.patience; bad=0

    def run_epoch(Xn, yn, train=True, batch=256):
        model.train(mode=train)
        losses=[]; n = Xn.shape[0]
        idx = np.random.permutation(n) if train else np.arange(n)
        for i in range(0, n, batch):
            j = idx[i:i+batch]
            xb = torch.from_numpy(Xn[j]).to(device)
            yb = torch.from_numpy(yn[j]).to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses))

    for ep in range(1, args.epochs+1):
        tr_loss = run_epoch(X_train, y_train, train=True, batch=args.batch)
        with torch.no_grad():
            model.eval()
            val_logits = model(torch.from_numpy(X_val).to(device))
            val_loss = float(crit(val_logits, torch.from_numpy(y_val).to(device)).item())
        print(f"Epoch {ep:02d} | train_loss {tr_loss:.4f} | val_loss {val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss; best_state = model.state_dict(); bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is None: best_state = model.state_dict()
    torch.save(best_state, args.out)
    print("Saved model to", args.out)

    # Final test metrics
    with torch.no_grad():
        model.load_state_dict(best_state); model.eval()
        logits = model(torch.from_numpy(X_test).to(device))
        pred = logits.argmax(1).cpu().numpy()
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=[0,1])
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, labels=[0,1], zero_division=0)
    metrics = {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "precision": {"not_help": float(prec[0]), "help": float(prec[1])},
        "recall":    {"not_help": float(rec[0]),  "help": float(rec[1])},
        "f1":        {"not_help": float(f1[0]),   "help": float(f1[1])},
    }
    print("Test metrics:", json.dumps(metrics, indent=2))

    df_metrics = pd.DataFrame({
        "Class": ["not_help", "help"],
        "Precision": [metrics["precision"]["not_help"], metrics["precision"]["help"]],
        "Recall":    [metrics["recall"]["not_help"], metrics["recall"]["help"]],
        "F1-score":  [metrics["f1"]["not_help"], metrics["f1"]["help"]]
    })

    cm = metrics["confusion_matrix"]
    df_cm = pd.DataFrame(cm,
                        index=["True Not Help", "True Help"],
                        columns=["Pred Not Help", "Pred Help"])

    df_metrics = pd.DataFrame({
        "Class": ["not_help", "help"],
        "Precision": [metrics["precision"]["not_help"], metrics["precision"]["help"]],
        "Recall":    [metrics["recall"]["not_help"],  metrics["recall"]["help"]],
        "F1-score":  [metrics["f1"]["not_help"],      metrics["f1"]["help"]]
    })

    df_summary = pd.DataFrame({
        "Metric": ["Accuracy"],
        "Value": [metrics["accuracy"]]
    })

    with pd.ExcelWriter("performance_evaluation.xlsx") as writer:
        df_metrics.to_excel(writer, sheet_name="Class Metrics", index=False)
        df_cm.to_excel(writer, sheet_name="Confusion Matrix", index=True)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)


# =========================
# Subcommand: infer (live)
# =========================
def cmd_infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=FEAT_DIM, n_classes=2, p_drop=0.0).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           model_complexity=1, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    hold = 0; last_fire = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.flip: frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        feat63=None
        if res.multi_hand_landmarks:
            feat63 = landmarks_to_63(res.multi_hand_landmarks[0].landmark)

        H,W = frame.shape[:2]
        if feat63 is not None:
            x = torch.from_numpy(feat63[None,:]).float().to(device)
            with torch.no_grad():
                prob = torch.softmax(model(x), dim=1)[0].cpu().numpy()  # [2]
            p_not, p_help = float(prob[0]), float(prob[1])

            cv2.putText(frame, f"not_help: {p_not:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(200,200,200),2)
            cv2.putText(frame, f"help:     {p_help:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),2)

            if p_help >= args.thresh:
                hold += 1
            else:
                hold = 0

            if hold >= args.hold:
                last_fire = time.time(); hold = 0
                cv2.putText(frame, "TRIGGER: HELP", (10, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,0,255),2)
        else:
            cv2.putText(frame, "No hand detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2)

        if time.time() - last_fire < 1.0:
            cv2.rectangle(frame, (0,0), (W,40), (0,0,255), -1)
            cv2.putText(frame, "HELP DETECTED", (14,28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,255,255),2)

        cv2.imshow("infer_singleframe", frame)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]: break

    hands.close(); cap.release(); cv2.destroyAllWindows()

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Single-frame MLP pipeline (63 features, binary).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="Collect labeled single-frame samples")
    c.add_argument("--outdir", default="data_single")
    c.add_argument("--width", type=int, default=1280)
    c.add_argument("--height", type=int, default=720)
    c.add_argument("--flip", action="store_true")
    c.set_defaults(func=cmd_collect)

    t = sub.add_parser("train", help="Train MLP on 63-d features")
    t.add_argument("--index", default="data_single/index_single.csv")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--patience", type=int, default=6)
    t.add_argument("--out", default="model_mlp_single.pth")
    t.set_defaults(func=cmd_train)

    i = sub.add_parser("infer", help="Live inference with trained MLP")
    i.add_argument("--model", default="model_mlp_single.pth")
    i.add_argument("--width", type=int, default=1280)
    i.add_argument("--height", type=int, default=720)
    i.add_argument("--flip", action="store_true")
    i.add_argument("--thresh", type=float, default=0.7)
    i.add_argument("--hold", type=int, default=6)
    i.set_defaults(func=cmd_infer)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
