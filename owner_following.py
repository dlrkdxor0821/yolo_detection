import argparse
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


BUTTON_X1, BUTTON_Y1 = 16, 16
BUTTON_X2, BUTTON_Y2 = 176, 60
REGISTER_SECONDS_FRONT = 3.0
REGISTER_SECONDS_BACK = 3.0
MIN_BACK_SAMPLES = 8
YOLO_PERSON_CLASS = 0
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-8 or bn < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def compute_hsv_hist(image_bgr: np.ndarray, bins_h: int = 32, bins_s: int = 32) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins_h, bins_s], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def compute_face_descriptor(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def extract_clothes_roi(person_crop: np.ndarray, face_box_local: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    h, w = person_crop.shape[:2]
    if h < 20 or w < 20:
        return person_crop

    top = int(h * 0.22)
    bottom = int(h * 0.78)
    left = int(w * 0.12)
    right = int(w * 0.88)
    clothes = person_crop[top:bottom, left:right].copy()

    if face_box_local is not None and clothes.size > 0:
        fx1, fy1, fx2, fy2 = face_box_local
        fx1 = clamp(fx1 - left, 0, max(0, right - left - 1))
        fx2 = clamp(fx2 - left, 0, max(0, right - left - 1))
        fy1 = clamp(fy1 - top, 0, max(0, bottom - top - 1))
        fy2 = clamp(fy2 - top, 0, max(0, bottom - top - 1))
        if fx2 > fx1 and fy2 > fy1:
            cv2.rectangle(clothes, (fx1, fy1), (fx2, fy2), (0, 0, 0), thickness=-1)
    return clothes


@dataclass
class PersonCandidate:
    box_xyxy: Tuple[int, int, int, int]
    conf: float
    face_box_local: Optional[Tuple[int, int, int, int]]
    face_desc: Optional[np.ndarray]
    clothes_hist: np.ndarray
    combined_score: float = 0.0
    face_score: float = 0.0
    clothes_score: float = 0.0


class OwnerRecognizer:
    def __init__(self, yolo_model: str, conf_thres: float, owner_threshold: float, face_model: str):
        self.detector = YOLO(yolo_model)
        self.conf_thres = conf_thres
        self.owner_threshold = owner_threshold
        self.face_detector_yunet = self._init_yunet(face_model)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.owner_face_desc: Optional[np.ndarray] = None
        self.owner_clothes_hist: Optional[np.ndarray] = None
        self.owner_registered = False

        self.registering = False
        self.register_phase = "idle"
        self.register_phase_started_at = 0.0
        self.awaiting_back_registration = False
        self.auto_full_registration = False
        self.register_face_samples: List[np.ndarray] = []
        self.register_clothes_front: List[np.ndarray] = []
        self.register_clothes_back: List[np.ndarray] = []

    def _init_yunet(self, model_path: str):
        model_file = Path(model_path)
        if not model_file.exists():
            model_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(YUNET_MODEL_URL, str(model_file))
            except Exception:
                return None

        if not hasattr(cv2, "FaceDetectorYN"):
            return None
        try:
            return cv2.FaceDetectorYN.create(str(model_file), "", (320, 320), 0.7, 0.3, 5000)
        except Exception:
            return None

    def start_front_registration(self, *, auto_continue_back: bool = False) -> None:
        self.registering = True
        self.register_phase = "front"
        self.register_phase_started_at = time.time()
        self.awaiting_back_registration = False
        self.auto_full_registration = auto_continue_back
        self.owner_registered = False
        self.register_face_samples = []
        self.register_clothes_front = []
        self.register_clothes_back = []

    def start_back_registration(self) -> bool:
        if not self.register_clothes_front:
            return False
        self.registering = True
        self.register_phase = "back"
        self.register_phase_started_at = time.time()
        self.awaiting_back_registration = False
        return True

    def get_registration_guide(self, now_ts: float) -> str:
        if self.register_phase == "front":
            remain = max(0.0, REGISTER_SECONDS_FRONT - (now_ts - self.register_phase_started_at))
            return f"Front phase: look at camera ({remain:.1f}s)"
        if self.register_phase == "back":
            remain = max(0.0, REGISTER_SECONDS_BACK - (now_ts - self.register_phase_started_at))
            return f"Back phase: show your back ({remain:.1f}s)"
        return "Click button to start registration."

    def update_registration(self, target: Optional[PersonCandidate], now_ts: float) -> str:
        if not self.registering:
            return "idle"

        if self.register_phase == "front":
            if target is not None:
                if target.face_desc is not None:
                    self.register_face_samples.append(target.face_desc)
                self.register_clothes_front.append(target.clothes_hist)

            if now_ts - self.register_phase_started_at >= REGISTER_SECONDS_FRONT:
                if self.auto_full_registration and self.register_clothes_front:
                    self.register_phase = "back"
                    self.register_phase_started_at = now_ts
                    self.awaiting_back_registration = False
                    return "auto_to_back"
                self.auto_full_registration = False
                self.registering = False
                self.register_phase = "idle"
                self.awaiting_back_registration = True
                return "front_done"
            return "collecting_front"

        if target is not None:
            self.register_clothes_back.append(target.clothes_hist)

        if now_ts - self.register_phase_started_at >= REGISTER_SECONDS_BACK:
            if len(self.register_clothes_back) < MIN_BACK_SAMPLES:
                self.auto_full_registration = False
                self.registering = False
                self.register_phase = "idle"
                self.awaiting_back_registration = True
                return "back_missing"
            self.finalize_registration()
            return "all_done"
        return "collecting_back"

    def detect_face_in_person(self, person_crop: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        if self.face_detector_yunet is not None:
            h, w = person_crop.shape[:2]
            self.face_detector_yunet.setInputSize((w, h))
            _, faces = self.face_detector_yunet.detect(person_crop)
            if faces is not None and len(faces) > 0:
                # Face format: [x, y, w, h, landmarks..., score]
                best = max(faces, key=lambda f: float(f[2] * f[3] * f[-1]))
                x, y, fw, fh = best[:4]
                x1 = clamp(int(x), 0, w - 1)
                y1 = clamp(int(y), 0, h - 1)
                x2 = clamp(int(x + fw), 0, w - 1)
                y2 = clamp(int(y + fh), 0, h - 1)
                if x2 > x1 and y2 > y1:
                    face_crop = person_crop[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        return (x1, y1, x2, y2), compute_face_descriptor(face_crop)

        # Fallback to Haar if YuNet is unavailable.
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None, None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        x2, y2 = x + w, y + h
        face_crop = person_crop[y:y2, x:x2]
        if face_crop.size == 0:
            return None, None
        face_desc = compute_face_descriptor(face_crop)
        return (x, y, x2, y2), face_desc

    def detect_people(self, frame: np.ndarray) -> List[PersonCandidate]:
        results = self.detector.predict(frame, conf=self.conf_thres, verbose=False)
        if not results:
            return []

        candidates: List[PersonCandidate] = []
        boxes = results[0].boxes
        if boxes is None:
            return []

        h, w = frame.shape[:2]
        for b in boxes:
            cls_id = int(b.cls[0].item())
            if cls_id != YOLO_PERSON_CLASS:
                continue

            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            x1, y1 = clamp(x1, 0, w - 1), clamp(y1, 0, h - 1)
            x2, y2 = clamp(x2, 0, w - 1), clamp(y2, 0, h - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            face_box_local, face_desc = self.detect_face_in_person(crop)
            clothes_roi = extract_clothes_roi(crop, face_box_local)
            if clothes_roi.size == 0:
                continue
            clothes_hist = compute_hsv_hist(clothes_roi)
            conf = float(b.conf[0].item())

            candidates.append(
                PersonCandidate(
                    box_xyxy=(x1, y1, x2, y2),
                    conf=conf,
                    face_box_local=face_box_local,
                    face_desc=face_desc,
                    clothes_hist=clothes_hist,
                )
            )
        return candidates

    def choose_registration_target(self, candidates: List[PersonCandidate], frame_w: int, frame_h: int) -> Optional[PersonCandidate]:
        if not candidates:
            return None
        cx, cy = frame_w / 2.0, frame_h / 2.0

        def score(c: PersonCandidate) -> float:
            x1, y1, x2, y2 = c.box_xyxy
            area = (x2 - x1) * (y2 - y1)
            bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = math.hypot(bx - cx, by - cy)
            return area - 2.2 * dist

        return max(candidates, key=score)

    def finalize_registration(self) -> None:
        cloth_hists = self.register_clothes_front + self.register_clothes_back
        if not cloth_hists:
            self.registering = False
            self.auto_full_registration = False
            return

        self.owner_clothes_hist = np.mean(np.stack(cloth_hists, axis=0), axis=0).astype(np.float32)
        self.owner_clothes_hist = self.owner_clothes_hist / (np.linalg.norm(self.owner_clothes_hist) + 1e-8)

        if self.register_face_samples:
            self.owner_face_desc = np.mean(np.stack(self.register_face_samples, axis=0), axis=0).astype(np.float32)
            self.owner_face_desc = self.owner_face_desc / (np.linalg.norm(self.owner_face_desc) + 1e-8)
        else:
            self.owner_face_desc = None

        self.owner_registered = True
        self.registering = False
        self.register_phase = "idle"
        self.awaiting_back_registration = False
        self.auto_full_registration = False

    def score_candidates(self, candidates: List[PersonCandidate]) -> Optional[PersonCandidate]:
        if not self.owner_registered or self.owner_clothes_hist is None:
            return None

        best: Optional[PersonCandidate] = None
        best_score = -1.0

        for c in candidates:
            cloth_score = cosine_similarity(self.owner_clothes_hist, c.clothes_hist)
            if c.face_desc is not None and self.owner_face_desc is not None:
                face_score = cosine_similarity(self.owner_face_desc, c.face_desc)
                combined = 0.35 * face_score + 0.65 * cloth_score
            else:
                face_score = 0.0
                combined = 0.70 * cloth_score

            c.face_score = face_score
            c.clothes_score = cloth_score
            c.combined_score = combined

            if combined > best_score:
                best_score = combined
                best = c

        if best is None or best.combined_score < self.owner_threshold:
            return None
        return best


class ClickState:
    def __init__(self) -> None:
        self.register_clicked = False


def on_mouse(event, x, y, _flags, click_state: ClickState):
    if event == cv2.EVENT_LBUTTONDOWN:
        if BUTTON_X1 <= x <= BUTTON_X2 and BUTTON_Y1 <= y <= BUTTON_Y2:
            click_state.register_clicked = True


def draw_button(frame: np.ndarray, active: bool) -> None:
    color = (30, 170, 255) if active else (70, 70, 70)
    cv2.rectangle(frame, (BUTTON_X1, BUTTON_Y1), (BUTTON_X2, BUTTON_Y2), color, thickness=-1)
    cv2.rectangle(frame, (BUTTON_X1, BUTTON_Y1), (BUTTON_X2, BUTTON_Y2), (220, 220, 220), thickness=2)
    text = "Register Owner"
    cv2.putText(frame, text, (BUTTON_X1 + 16, BUTTON_Y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def handle_register_action(recognizer: OwnerRecognizer) -> str:
    if recognizer.registering:
        return "Registration is in progress."
    if recognizer.awaiting_back_registration:
        ok_start = recognizer.start_back_registration()
        if ok_start:
            return "Back registration started: show your back for 3s."
        return "Front samples missing. Start front registration first."
    recognizer.start_front_registration(auto_continue_back=False)
    return "Front registration started: look at camera for 3s."


def main():
    parser = argparse.ArgumentParser(description="Owner recognition with face + clothes features.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model path/name.")
    parser.add_argument(
        "--face-model",
        default="models/face_detection_yunet_2023mar.onnx",
        help="YuNet face detector model path (.onnx).",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--owner-threshold", type=float, default=0.55, help="Owner matching threshold.")
    args = parser.parse_args()

    recognizer = OwnerRecognizer(
        yolo_model=args.model,
        conf_thres=args.conf,
        owner_threshold=args.owner_threshold,
        face_model=args.face_model,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    win_name = "Owner Following"
    cv2.namedWindow(win_name)
    click_state = ClickState()
    cv2.setMouseCallback(win_name, on_mouse, click_state)

    status = "Click button or Space: register owner (front, then back)."
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if click_state.register_clicked:
            status = handle_register_action(recognizer)
            click_state.register_clicked = False

        candidates = recognizer.detect_people(frame)
        frame_h, frame_w = frame.shape[:2]

        if recognizer.registering:
            now_ts = time.time()
            target = recognizer.choose_registration_target(candidates, frame_w, frame_h)
            if target is not None:
                x1, y1, x2, y2 = target.box_xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 3)
            reg_state = recognizer.update_registration(target, now_ts)
            status = recognizer.get_registration_guide(now_ts)
            if reg_state == "auto_to_back":
                status = "Front done (auto). Turn around: back 3s."
            elif reg_state == "front_done":
                status = "Front done. Button or Space again for back (3s)."
            elif reg_state == "all_done":
                status = "Owner registration complete."
            elif reg_state == "back_missing":
                status = (
                    f"Back samples too low (<{MIN_BACK_SAMPLES}). "
                    "Button or Space again to retry back 3s."
                )
            elif target is None:
                status = "No clear target. Keep one person at center."

        owner = recognizer.score_candidates(candidates)
        for c in candidates:
            x1, y1, x2, y2 = c.box_xyxy
            if owner is not None and c is owner:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 240, 60), 3)
                label = f"OWNER {owner.combined_score:.2f} (F:{owner.face_score:.2f} C:{owner.clothes_score:.2f})"
                cv2.putText(frame, label, (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 240, 60), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 50, 230), 2)
                cv2.putText(frame, "person", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 50, 230), 2)

        draw_button(frame, recognizer.registering)
        cv2.putText(frame, status, (18, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2)
        cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            status = handle_register_action(recognizer)
        if key == ord("r"):
            status = handle_register_action(recognizer)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
