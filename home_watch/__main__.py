import argparse
import json
from pathlib import Path
import sys
import cv2
from .models import download
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
    commands.add_parser("download-models")
    for name in ("compare", "webcam"):
        command = commands.add_parser(name)
        command.add_argument("--reference", type=Path, required=True)
        command.add_argument("--familiar-threshold", type=float, default=0.45)
        command.add_argument("--unfamiliar-threshold", type=float, default=0.30)
        if name == "compare":
            command.add_argument("--image", type=Path, required=True)
        else:
            command.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.command == "download-models":
            download(args.models)
            return 0
        matcher = FaceMatcher(args.models, Thresholds(args.unfamiliar_threshold, args.familiar_threshold))
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
