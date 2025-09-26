# collect_sequences.py
# pip install mediapipe opencv-python numpy
import cv2, time, os, argparse, uuid
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

# 21 joints per hand, we’ll use (x,y,z) normalized, both hands concatenated + masks
NUM_JOINTS = 21
FEAT_PER_HAND = 3 * NUM_JOINTS  # x,y,z
PAD_ZERO = np.zeros((NUM_JOINTS, 3), dtype=np.float32)

def norm_palm_frame(lm):
    """Return landmarks in a palm-centric, scale-normalized frame (Nx3)."""
    L = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # normalized image coords
    WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0,5,9,17

    O = L[WRIST]
    a = L[INDEX_MCP] - O
    b = L[PINKY_MCP] - O
    u = b - a; un = np.linalg.norm(u); u = u/un if un>1e-8 else np.array([1,0,0],np.float32)
    v = L[MIDDLE_MCP] - O; v = v - (u@v)*u; vn = np.linalg.norm(v); v = v/vn if vn>1e-8 else np.array([0,1,0],np.float32)
    w = np.cross(u,v); wn = np.linalg.norm(w); w = w/wn if wn>1e-8 else np.array([0,0,1],np.float32)
    R = np.stack([u,v,w], axis=1)
    X = (L - O) @ R
    scale = np.linalg.norm(((L[MIDDLE_MCP]-O) @ R)[:2]) + 1e-6
    Xn = X/scale

    # flip w so MCP row tends to positive, making “into palm” negative
    mcp_idx = [5,9,13,17]
    if Xn[mcp_idx,2].mean() < 0:
        Xn[:,2] *= -1.0
    return Xn

def handpack(frame_hands):
    """
    Pack both hands into a fixed-length feature:
    [Left(21x3), Right(21x3), left_present, right_present] -> (21*3*2 + 2,)
    If only one hand exists, the other is zeros with present=0.
    """
    # Determine which is Left/Right from handedness
    hands_sorted = {"Left": None, "Right": None}
    for lm, handedness in frame_hands:
        label = handedness.classification[0].label  # 'Left' or 'Right'
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
    F = np.concatenate(feats, axis=0).reshape(-1)  # (21*3*2,)
    F = np.concatenate([F, np.array(mask, dtype=np.float32)])  # +2 mask bits
    return F  # shape: 21*3*2 + 2 = 128

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data", help="output folder for .npz and index.csv")
    ap.add_argument("--seq_len", type=int, default=48, help="frames per sequence window")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--flip", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    index_path = os.path.join(args.outdir, "index.csv")
    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write("path,label\n")

    label_map = {
        "h": 1,   # help gesture
        "o": 2,   # open palm
        "f": 3,   # fist
        "w": 4,   # wave / other
        "n": 0,   # negative / background
    }
    print("Keys: [h]=help, [o]=open, [f]=fist, [w]=wave, [n]=negative, [q]=quit")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           model_complexity=1, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    buf = []

    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.flip: frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        pairs = []
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                pairs.append((lm, hd))

        feat = handpack(pairs)  # 128-dim
        buf.append(feat)
        if len(buf) > args.seq_len:
            buf.pop(0)

        # HUD
        cv2.putText(frame, f"buffer: {len(buf)}/{args.seq_len}", (12,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, "label keys: [h]=help [o]=open [f]=fist [w]=wave [n]=neg [q]=quit",
                    (12,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1,cv2.LINE_AA)
        cv2.imshow("collect", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27: break
        for key, lbl in label_map.items():
            if k == ord(key) and len(buf) == args.seq_len:
                arr = np.stack(buf, axis=0)  # [T, 128]
                fname = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}-L{lbl}.npz"
                path = os.path.join(args.outdir, fname)
                np.savez_compressed(path, seq=arr, label=np.int64(lbl))
                with open(index_path, "a") as f:
                    f.write(f"{path},{lbl}\n")
                print("saved:", path)

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
