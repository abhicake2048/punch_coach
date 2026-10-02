"""Smoke-test tracked detection and cropped pose on every labeled test video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from core.boxer_pipeline import TrackedBoxerPosePipeline
from core.pose_engine import PoseEngine


def validate_video(
    path: Path,
    pipeline: TrackedBoxerPosePipeline,
    frame_limit: int,
) -> tuple[int, int, int, set[int]]:
    """Return readable, tracked, pose frame counts and observed tracker IDs."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0, 0, 0, set()
    pipeline.reset()
    readable = tracked = posed = 0
    track_ids: set[int] = set()
    try:
        while readable < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            stages = pipeline.process(frame)
            readable += 1
            if stages.tracked_box is not None:
                tracked += 1
                track_ids.add(stages.tracked_box.track_id)
            if stages.raw_keypoints_original is not None:
                posed += 1
    finally:
        capture.release()
    return readable, tracked, posed, track_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("test_files"))
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")

    videos = sorted(args.directory.glob("*_punches.mp4"))
    if not videos:
        parser.error(f"no *_punches.mp4 videos found in {args.directory}")
    engine = PoseEngine(weights="yolo11s-pose.pt", confidence_threshold=0.25)
    pipeline = TrackedBoxerPosePipeline(engine, detector_weights="yolo11s-pose.pt")

    failed = False
    print("video,readable,tracked,posed,track_ids,status")
    for video in videos:
        readable, tracked, posed, track_ids = validate_video(video, pipeline, args.frames)
        status = "PASS" if readable == args.frames and tracked > 0 and posed > 0 else "FAIL"
        failed |= status == "FAIL"
        ids = ";".join(str(value) for value in sorted(track_ids))
        print(f"{video.name},{readable},{tracked},{posed},{ids},{status}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
