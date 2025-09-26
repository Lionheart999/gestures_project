# infer_live.py
# pip install mediapipe opencv-python numpy torch
import cv2, argparse, numpy as np, torch, torch.nn as nn, time
import mediapipe as mp

from collections import deque

mp_hands = mp.solutions.hands
NUM_JOINTS = 21
FEAT_DIM = 128        # from collect script (21*3*2 + 2)
SEQ_LEN = 48
CLASSES = ["neg","help","open","fist","wave"]

class TinyTCN(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 96, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 96, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Linear(96, n_classes)
    def forward(self, x):         # [B,T,F]
        x = x.transpose(1,2)      # [B,F,T]
        x = self.net(x).squeeze(-1)
        return self.head(x)

# --- copy normalization and packers from collector ---
def norm_palm_frame(lm):
    L = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0,5,9,17
    O = L[WRIST]
    a = L[INDEX_MCP] - O
    b = L[PINKY_MCP] - O
    u = b - a; un=np.linalg.norm(u); u=u/un if un>1e-8 else np.array([1,0,0],np.float32)
    v = L[MIDDLE_MCP] - O; v=v-(u@v)*u; vn=np.linalg.norm(v); v=v/vn if vn>1e-8 else np.array([0,1,0],np.float32)
    w = np.cross(u,v); wn=np.linalg.norm(w); w=w/wn if wn>1e-8 else np.array([0,0,1],np.float32)
    R = np.stack([u,v,w], axis=1)
    X = (L - O) @ R
    scale = np.linalg.norm(((L[MIDDLE_MCP]-O) @ R)[:2]) + 1e-6
    Xn = X/scale
    mcp_idx = [5,9,13,17]
    if Xn[mcp_idx,2].mean() < 0: Xn[:,2]*=-1.0
    return Xn

PAD_ZERO = np.zeros((NUM_JOINTS,3), np.float32)
def handpack(pairs):
    hands_sorted = {"Left": None, "Right": None}
    for lm, hd in pairs:
        label = hd.classification[0].label
        hands_sorted[label] = lm
    feats=[]; mask=[]
    for side in ["Left","Right"]:
        if hands_sorted[side] is None:
            feats.append(PAD_ZERO); mask.append(0.0)
        else:
            Xn = norm_palm_frame(hands_sorted[side].landmark)
            feats.append(Xn); mask.append(1.0)
    F = np.concatenate(feats, axis=0).reshape(-1)
    F = np.concatenate([F, np.array(mask, dtype=np.float32)])
    return F  # (128,)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_tcn.pth")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--thresh", type=float, default=0.70)
    ap.add_argument("--hold", type=int, default=6, help="frames over thresh to trigger")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTCN(n_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           model_complexity=1, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    window = deque(maxlen=SEQ_LEN)
    over_cnt = 0; last_class = None; last_fire = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.flip: frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        pairs=[]
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                pairs.append((lm, hd))

        feat = handpack(pairs)
        window.append(feat)
        H,W = frame.shape[:2]

        if len(window) == SEQ_LEN:
            X = torch.from_numpy(np.stack(window)[None,:,:]).float().to(device)  # [1,T,F]
            with torch.no_grad():
                prob = torch.softmax(model(X), dim=1)[0].cpu().numpy()  # [C]
            # display
            y = 20
            for i,c in enumerate(CLASSES):
                cv2.putText(frame, f"{c}: {prob[i]:.2f}", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if c!="neg" else (200,200,200),2)
                y += 28
            c_idx = int(prob.argmax())
            c_name = CLASSES[c_idx]
            if c_name != "neg" and prob[c_idx] >= args.thresh:
                if last_class == c_idx:
                    over_cnt += 1
                else:
                    over_cnt = 1
                last_class = c_idx
            else:
                over_cnt = 0; last_class = None

            if over_cnt >= args.hold:
                last_fire = time.time()
                over_cnt = 0
                cv2.putText(frame, f"TRIGGER: {c_name}", (10, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # fade banner after trigger
        if time.time() - last_fire < 1.0:
            cv2.rectangle(frame, (0,0), (W,60), (0,0,255), -1)
            cv2.putText(frame, "GESTURE DETECTED", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255),3)

        cv2.imshow("infer_live", frame)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]: break

    hands.close(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
