# yolo_detection

얼굴 + 옷 특징을 함께 사용해서 주인을 판별하고, `주인인식` 버튼으로 주인을 등록하는 데모입니다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

```bash
python3 owner_following.py --model yolov8n.pt
```

옵션:

- `--model`: YOLO 가중치 (`yolov8n.pt`, `yolov8s.pt` 등)
- `--face-model`: YuNet 얼굴 검출 ONNX 경로 (기본 `models/face_detection_yunet_2023mar.onnx`)
- `--camera`: 카메라 인덱스 (기본 `0`)
- `--conf`: 사람 검출 confidence threshold (기본 `0.35`)
- `--owner-threshold`: 주인 판별 임계치 (기본 `0.55`)

## 사용 방법

- 등록은 **좌측 상단 버튼** 또는 **`Space`(스페이스)** 로 동일하게 진행
- 1차: 앞모습 3초 등록
- 2차: 뒷모습 3초 등록
- 뒷모습에서 사람이 충분히 검출되지 않으면 재시도 안내가 뜸
- 등록 완료 후 다중 인원에서 주인만 `OWNER`로 초록 박스 표시
- `q`: 종료
- `r`: 버튼/`Space`와 동일한 수동 등록 트리거(앞 -> 다시 입력 -> 뒤)
- 버튼/`Space`는 여전히 **앞 → (다시) 뒤** 2단계 수동 등록

## 구현 요약

- 사람 검출: YOLO (`person` 클래스)
- 얼굴 검출: YuNet (실패 시 Haar cascade fallback)
- 얼굴 특징: 얼굴 ROI grayscale histogram
- 옷 특징: 상반신 ROI HSV histogram
- 최종 점수: 얼굴 점수 + 옷 점수 가중합(combined = 0.35 * face_score + 0.65 * cloth_score)

## 참고

- 첫 실행 시 YuNet 모델 파일이 없으면 자동 다운로드합니다.