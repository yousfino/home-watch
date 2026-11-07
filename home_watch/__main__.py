import argparse
import json
from pathlib import Path
import sys
import cv2
from .models import download, MODELS, verified
from .detection import RetinaFace
from .recognition import FaceMatcher, Thresholds


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def main():
    parser = argparse.ArgumentParser(description="Home Watch local face comparison")
    parser.add_argument("--models", type=Path, default=Path("models"))
    commands = parser.add_subparsers(dest="command", required=True)
    model_command = commands.add_parser("download-models")
    model_command.add_argument("--detector-only", action="store_true")
    for name in ("compare", "webcam", "detect"):
        command = commands.add_parser(name)
        command.add_argument("--detection-confidence", type=float, default=0.7)
        command.add_argument("--detector-size", type=int, choices=(640, 960, 1280), default=640)
        if name != "detect":
            command.add_argument("--reference", type=Path, required=True)
        if name != "detect":
            command.add_argument("--familiar-threshold", type=float, default=0.45)
            command.add_argument("--unfamiliar-threshold", type=float, default=0.30)
        if name in ("compare", "detect"):
            command.add_argument("--image", type=Path, required=True)
        else:
            command.add_argument("--camera", type=int, default=0)
        if name == "detect":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "download-models":
            download(args.models, detector_only=args.detector_only)
            return 0
        if args.command == "detect":
            name = "retinaface_r34.onnx"
            if not verified(args.models / name, MODELS[name][1]):
                raise ValueError("Missing or invalid RetinaFace model. Run download-models first.")
            if args.output.resolve() == args.image.resolve():
                raise ValueError("Diagnostic output must not overwrite the input image")
            image = read_image(args.image)
            detector = RetinaFace(args.models / name, args.detection_confidence, args.detector_size)
            faces = detector.detect(image)
            details = []
            for face in faces:
                x, y, w, h = map(int, face[:4])
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 200, 0), 2)
                cv2.putText(image, f"{face[-1]:.3f}", (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 200, 0), 2)
                for px, py in face[4:14].reshape(5, 2):
                    cv2.circle(image, (int(px), int(py)), 2, (0, 0, 255), -1)
                details.append({"box": face[:4].tolist(), "confidence": float(face[-1]), "landmarks": face[4:14].reshape(5, 2).tolist()})
            if not cv2.imwrite(str(args.output), image):
                raise ValueError(f"Cannot save diagnostic image: {args.output}")
            print(json.dumps({"faces": details, "output": str(args.output)}, indent=2))
            return 0
        matcher = FaceMatcher(args.models, Thresholds(args.unfamiliar_threshold, args.familiar_threshold),
                              args.detection_confidence, args.detector_size)
        reference = matcher.enroll(read_image(args.reference))
        if args.command == "compare":
            print(json.dumps({"faces": matcher.compare(read_image(args.image), reference)}, indent=2))
            return 0
        capture = cv2.VideoCapture(args.camera)
        try:
            if not capture.isOpened():
                raise ValueError("Cannot open webcam. Check camera permissions and close other camera apps.")
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Webcam stopped returning frames")
                for result in matcher.compare(frame, reference):
                    x, y, w, h = result["box"]
                    color = {"Familiar": (0, 200, 0), "Unfamiliar": (0, 0, 255), "Unresolved": (0, 200, 255)}[result["label"]]
                    text = result["label"]
                    if result["similarity"] is not None:
                        text += f' {result["similarity"]:.3f}'
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, text, (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow("Home Watch - Q to quit", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
        return 0
    except (ValueError, OSError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
