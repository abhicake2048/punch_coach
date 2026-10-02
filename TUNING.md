# CornerCoach calibration guide

Punch counting and punch classification are intentionally separate. Every
validated outward-extension event increments the punch count, even when its
type is `Unclassified`. Calibrate the count first; only then tune the angle and
trajectory rules used for Jab, Cross, Hook, and Uppercut labels.

## 1. Create ground truth

Watch each calibration clip frame by frame and record at least:

- total punch count;
- approximate punch timestamps;
- hand, when visible;
- uncertain or occluded punches.

A single total can tune count error, but timestamps are needed to distinguish
false positives from missed punches. Use several clips and reserve at least one
clip as a holdout test so parameters are not overfit to one boxer or camera.

## 2. Extract diagnostics once

```powershell
python test_cli.py ".\test_files\4804863-uhd_3840_2160_25fps.mp4" `
  --seconds 30 `
  --expected-count 18 `
  --diagnostics-csv ".\test_files\diagnostics_11s.csv"
```

The CSV contains time, hand, One-Euro-filtered torso-relative wrist
speed, outward extension speed, reach, elbow angle, dynamic threshold,
detector phase, and detections.
Plot these columns or filter `detected=True` to audit each event.

Regenerate diagnostics whenever the filter, detector/pose weights, confidence
threshold, or crop geometry changes. CSVs produced by an older signal pipeline
must not be mixed into a new calibration.

## 3. Replay a parameter grid without rerunning YOLO

```powershell
python calibrate_detector.py `
  ".\test_files\diagnostics_11s.csv=18" `
  ".\test_files\diagnostics_15s.csv=45" `
  --min-speeds "0.3,0.4,0.5" `
  --extension-speeds "0.1,0.15,0.2" `
  --extension-gains "0.03,0.04,0.05" `
  --retraction-gains "0.02,0.03,0.04" `
  --refractory-frames "8,9,10" `
  --max-extension-gains "1.3,1.4,1.5" `
  --max-extension-velocities "20,25" `
  --min-outward-frames "2" `
  --min-count-angles "40,45,50"
```

The closest counts are printed first. Verify their timestamps against the
ground-truth list; matching the total alone can hide equal numbers of false
positives and false negatives.

## Punch-count parameters

| Parameter | Lower value | Higher value |
|---|---|---|
| Minimum wrist speed | More sensitivity and false positives | Misses slower punches |
| Minimum outward speed | Counts smaller outward motions | Requires clearer extension |
| Minimum reach gain | Accepts short/partial punches | Rejects small pose jitter |
| Partial retraction required | Re-arms after a short pullback | Requires a deeper reset |
| Minimum outward-motion frames | More responsive to short punches | Rejects short pose jumps |
| Minimum same-hand cycle gap | Allows faster repeat punches | Suppresses duplicate cycles |
| Maximum re-arm wait | Recovers sooner after lost tracking | Reduces duplicate risk after occlusion |
| Minimum cycle elbow angle | Accepts tightly bent motion | Rejects implausibly folded-arm cycles |
| Maximum reach gain/outward speed | Permits extreme motion | Rejects more keypoint teleports |
| Pose-jump rejection speed | Rejects more tracking jumps | Permits faster detected motion |
| One-Euro minimum cutoff | Smoother but more lag | More responsive but more jitter |
| One-Euro beta | Less speed adaptation | Less lag during fast punches |

The current profile was selected jointly from tracked diagnostics for the
7-punch and 14-punch clips. It produced 5 and 14 counts respectively; that
known residual error is preferable to the single-clip fit that produced 7 and
37. It requires at least two outward-motion frames and a nine-frame same-hand
rearm gap; 12 frames remains the safety ceiling for lost pose tracking.

The count profile is: minimum wrist speed `0.30 TL/s`, outward speed `0.10
TL/s`, reach gain `0.04 TL`, retraction `0.02 TL`, maximum reach gain `1.40 TL`,
maximum outward speed `25 TL/s`, pose-jump limit `20 TL/s`, and minimum cycle
angle `20°`. `TL` means torso length.

Change one parameter family at a time. For rapid combinations, first lower the
partial-retraction requirement slightly; do not reduce minimum outward-motion
frames to one unless timestamp review proves that keypoint jumps are under
control. For isolated false events, inspect their CSV rows before raising a
global threshold.

For a single clip, use `tune_detector.py CSV --expected-count N`. For a useful
calibration, always finish with `calibrate_detector.py` across multiple labeled
clips and then verify the winning profile on a separate holdout video.

## Pose signal processing

Raw YOLO coordinates never feed velocity, angle, guard, or punch logic. Each
coordinate first passes through an adaptive One-Euro low-pass filter. Its
cutoff rises during fast motion to reduce lag and falls during near-stationary
periods to suppress network jitter. The COCO shoulder midpoint is used as a
neck proxy and subtracted from every point, then coordinates are divided by
the shoulder-to-hip midpoint distance (torso length). Smoothed original-frame
pixel coordinates are retained for drawing.

Detection runs on a selectable 480px or 640px square, aspect-preserving
letterbox. ByteTrack locks one person ID. The corresponding rectangle is
expanded by 10% on every side, clipped to the original frame, and then cut from
the original pixels. That crop is letterboxed to the same selected scalar size,
and only the padded crop is sent to YOLO11 pose.

## Punch-type parameters

These do not change the total count:

- Jab/Cross minimum extension angle controls straight-punch labeling.
- Hook minimum/maximum angles define the bent-arm range.
- Hook horizontal ratio requires motion to be predominantly lateral.
- Uppercut maximum angle controls how bent the arm must be.
- Uppercut upward ratio requires a sufficiently upward image-plane trajectory.

After labeling punch types, evaluate a confusion matrix rather than only total
accuracy. Camera viewpoint strongly affects image-plane trajectory rules.

## Guard parameters

`Guard line: nose → shoulder` is an interpolation value:

- `0.0` places the threshold at nose height and is strict;
- `1.0` places it at shoulder height and is permissive;
- `0.65` is the default.

`Glove-cuff tolerance` shifts the line downward by a fraction of torso length.
Increase it when the pose model places the wrist at the glove cuff even
though the glove itself is protecting the face. The overall score is calculated
across eligible hand-frame observations, so one temporarily unobservable or
punching hand does not automatically fail the entire frame.

Review guard-drop start/recovery timestamps in `logs/cornercoach.log`. Do not
score frames where the relevant arm is punching, occluded, or in retraction.

## Fatigue parameters

The fatigue analyzer compares the first and final thirds of the clip. Tune the
work-rate and mean-peak-speed drop percentages only after punch count and speed
outliers are reliable. Short clips or segments with fewer than two early
punches should be treated as insufficient evidence rather than fatigue.

For serious evaluation, report count MAE, event precision/recall within a small
timestamp tolerance, per-type confusion matrices, and guard-frame precision
and recall across multiple boxers and camera views.
