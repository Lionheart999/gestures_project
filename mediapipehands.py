import cv2
import argparse
import time
import math
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def draw_bbox_and_size(image, landmarks, ih, iw):
    xs = [int(lm.x * iw) for lm in landmarks]
    ys = [int(lm.y * ih) for lm in landmarks]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
    # Approx “hand size” in pixels (diagonal of bbox)
    size = int(math.hypot(x2 - x1, y2 - y1))
    cv2.putText(image, f"hand_px~{size}", (x1, max(0,y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="0 for webcam, or path to video file")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--min_det", type=float, default=0.5,
                    help="min_detection_confidence")
    ap.add_argument("--min_track", type=float, default=0.5,
                    help="min_tracking_confidence")
    args = ap.parse_args()

    # Open source
    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    if args.source == "0":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_track
    )

    prev_t = time.time()
    fps_smooth = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ih, iw = frame.shape[:2]
        if res.multi_hand_landmarks:
            for hand_lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                # Draw bbox + approximate pixel size
                draw_bbox_and_size(frame, hand_lm.landmark, ih, iw)

                # Show handedness & confidence
                label = handedness.classification[0].label
                score = handedness.classification[0].score
                cv2.putText(frame, f"{label} {score:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_t) if now > prev_t else 0
        prev_t = now
        fps_smooth = fps if fps_smooth is None else (0.9*fps_smooth + 0.1*fps)
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, ih-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("MediaPipe Hands Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
