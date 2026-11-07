"""Commands for enrollment, independent evaluation and local prediction."""
import argparse
import json
from pathlib import Path
import sys
import cv2
from .detection import RetinaFace
from .models import MODELS, download, verified
from .profile import IdentityProfile
from .recognition import FaceEncoder
from .training import read_image, train, evaluate


def draw(image, results):
    for result in results:
        x, y, w, h = result['box']
        label = result.get('label', 'Face')
        color = (0, 200, 0) if label in ('Familiar', 'Face') else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        score = result.get('similarity', result['detection_confidence'])
        cv2.putText(image, f'{label} {score:.3f}', (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    return image


def save_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f'Cannot encode output image: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def main():
    parser = argparse.ArgumentParser(description='Home Watch: enroll one identity, detect and recognize faces')
    parser.add_argument('--models', type=Path, default=Path('models'))
    commands = parser.add_subparsers(dest='command', required=True)
    model_command = commands.add_parser('download-models')
    model_command.add_argument('--detector-only', action='store_true')
    for name in ('train', 'predict', 'webcam', 'evaluate', 'detect'):
        command = commands.add_parser(name)
        command.add_argument('--detection-confidence', type=float, default=.7)
        command.add_argument('--detector-size', type=int, choices=(640, 960, 1280), default=640)
        if name in ('predict', 'webcam', 'evaluate'):
            command.add_argument('--profile', type=Path, default=Path('data/identity-profile.npz'))
        if name == 'train':
            command.add_argument('--photos', type=Path, required=True)
            command.add_argument('--output', type=Path, default=Path('data/identity-profile.npz'))
            command.add_argument('--threshold', type=float, default=.4)
            command.add_argument('--seed', type=int, default=42)
        if name in ('predict', 'detect'):
            command.add_argument('--image', type=Path, required=True)
            command.add_argument('--output', type=Path, required=name == 'detect')
        if name == 'webcam':
            command.add_argument('--camera', type=int, default=0)
        if name == 'evaluate':
            command.add_argument('--familiar', type=Path, required=True)
            command.add_argument('--unfamiliar', type=Path, required=True)
            command.add_argument('--output', type=Path, default=Path('data/evaluation.json'))
    args = parser.parse_args()
    try:
        if args.command == 'download-models':
            download(args.models, args.detector_only)
            return 0
        if args.command in ('predict', 'detect') and args.output and args.output.resolve() == args.image.resolve():
            raise ValueError('Output must not overwrite the input image')
        if args.command == 'detect':
            model = args.models / 'retinaface_r34.onnx'
            if not verified(model, MODELS[model.name][1]):
                raise ValueError('Missing or invalid detector. Run download-models first')
            detector = RetinaFace(model, args.detection_confidence, args.detector_size)
            image = read_image(args.image)
            results = [{'box': f[:4].astype(int).tolist(), 'detection_confidence': float(f[-1])} for f in detector.detect(image)]
            save_image(args.output, draw(image, results))
            print(json.dumps({'faces': results, 'output': str(args.output)}, indent=2))
            return 0
        encoder = FaceEncoder(args.models, args.detection_confidence, args.detector_size)
        if args.command == 'train':
            report = train(encoder, args.photos, args.output, args.threshold, args.seed)
            print(json.dumps({k: report[k] for k in ('training_count_for_holdout', 'holdout_count',
                              'holdout_familiar_count', 'final_profile_photo_count', 'threshold_calibrated')}, indent=2))
            print(f'Profile: {args.output}; report: {args.output.with_suffix(".report.json")}')
            return 0
        profile = IdentityProfile.load(args.profile)
        if args.command == 'evaluate':
            report = evaluate(encoder, profile, args.familiar, args.unfamiliar)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))
            return 0
        if args.command == 'predict':
            image = read_image(args.image)
            results = encoder.predict(image, profile)
            print(json.dumps({'faces': results}, indent=2))
            if args.output:
                save_image(args.output, draw(image, results))
            return 0
        capture = cv2.VideoCapture(args.camera)
        try:
            if not capture.isOpened():
                raise ValueError('Cannot open camera. Check Windows camera permissions and close other camera apps')
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Camera stopped returning frames')
                cv2.imshow('Home Watch - Q to quit', draw(frame, encoder.predict(frame, profile)))
                if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
        return 0
    except (ValueError, OSError, KeyError, cv2.error) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
