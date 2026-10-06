# CornerCoach production inference

CornerCoach analyzes an uploaded boxing video on CPU with two-stage YOLO11 pose
and one trained ST-GCN. It reports punch counts, hand and punch type,
guard discipline, fatigue indicators, a metric-grounded Gemini coaching report,
an annotated video, and a downloadable PDF report.

## Start on Windows in three steps

Requirements: Windows 10/11, 64-bit Python 3.10-3.12, and the two weight files
already included under `weights/`.

1. Download or clone the repository, then open PowerShell in this `cornercoach`
   directory.

2. Create the private environment and install everything:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup.ps1
   ```

3. Start the app:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run.ps1
   ```

   To enable Gemini without saving the key to disk:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run.ps1 -GeminiApiKey "YOUR_KEY"
   ```

The browser opens at `http://localhost:8501`. Setup only needs to be repeated
when `requirements.txt` changes.

## Start on Linux or macOS in three steps

1. Download or clone the repository and open a terminal in `cornercoach`.

2. Run `bash setup.sh`.

3. Run `bash run.sh`, or enable Gemini for this launch with:

   ```bash
   GEMINI_API_KEY="YOUR_KEY" bash run.sh
   ```

The scripted local installation was chosen instead of a prebuilt Docker image
because the PyTorch/Ultralytics image would be several gigabytes, while these
scripts keep startup to three steps and use the correct native CPU wheel for the
host. No global Python packages are modified.

## Required production files

```text
app.py
core/
visualizer/
weights/yolo11s-pose.pt
weights/stgcn/best_checkpoint.pt
requirements.txt
setup.ps1 and run.ps1
setup.sh and run.sh
```

The setup scripts stop with a clear error if a required checkpoint is missing.

## Recommended CPU settings

The app always preserves the training-time top-down preprocessing order:

- YOLO11s first detects and tracks the primary person in the full frame.
- The tracked person box is expanded by 10% on every side.
- The padded boxer crop is letterboxed without stretching.
- A separate YOLO11s pose model extracts keypoints from that crop.
- Crop keypoints are mapped back into original-frame coordinates before pose
  smoothing, kinematics, and ST-GCN inference.

The CPU-safe runtime optimizations are:

- Full-frame person detection uses 320 pixels while the cropped pose stage stays
  at 480 pixels.
- Both YOLO stages run in ordered CPU batches of eight frames, reducing 640
  frames from roughly 1,280 individual model calls to about 160 batched calls.
- `Live diagnostic preview`: disabled. Every video frame is still analyzed and
  written to the final annotated video; Streamlit simply avoids expensive image
  and chart redraws during processing.
- Progress is refreshed every 30 frames instead of every frame.
- OpenCV uses one worker while PyTorch receives up to eight CPU threads, avoiding
  nested thread oversubscription.

Set a specific PyTorch CPU-thread count before starting if needed:

```powershell
$env:CORNERCOACH_CPU_THREADS = "4"
.\run.ps1
```

Measure speed on the actual production computer. More threads are not always
faster. The detector -> 10% padded crop -> pose sequence cannot be disabled in
the production interface because the existing checkpoints were trained from
features extracted with this top-down approach.

The application does not skip inference frames because the trained checkpoints
use 11-frame motion sequences and fast punches may last only 5-8 frames.

## Multiple users

The YOLO predictors, boxer tracker, and ST-GCN temporal state are shared to stay
within small cloud-memory limits, so a process-wide lease queue runs one upload
at a time in FIFO order. Waiting sessions do not block Python threads. Their page
refreshes the queue state once per second and shows both the live position and a
queue progress bar. Cached completed results and report viewing remain
session-specific and do not enter the inference queue.

Every queue entry belongs to its Streamlit browser session. Closing a waiting
tab removes that entry as soon as Streamlit reports the disconnect; a 20-second
heartbeat lease is the fallback if the runtime cannot report it. Closing the
tab that owns the active analysis requests cancellation. Video inference stops
after the current YOLO batch returns, temporary files and tracker state are
cleaned up, and only then is the next visitor allowed to start. H.264
transcoding is also terminated on disconnect. This ordering prevents a cancelled
job and its successor from using the shared models at the same time.

The worker lease is released from a nested `finally` block after success,
invalid media, cancellation, timeout, or an inference exception. A 45-minute
safety limit prevents an abnormal job from owning the queue indefinitely.
Streamlit Cloud defaults to batch size 4 and up to four PyTorch CPU threads;
larger hosts can select batch size 8 or set `CORNERCOACH_CPU_THREADS` explicitly.

The queue timings can be adjusted before launch if the hosting environment needs
different values:

```bash
export CORNERCOACH_QUEUE_POLL_SECONDS=1
export CORNERCOACH_QUEUE_LEASE_SECONDS=20
export CORNERCOACH_MAX_JOB_SECONDS=2700
```

The max-job limit and active-tab cancellation are checked at safe batch
boundaries. Python cannot safely interrupt a PyTorch/OpenCV native call in the
middle, so cancellation can take as long as the current batch rather than being
instantaneous.

## Using the app

1. Upload an MP4/MOV video or JPG/PNG image.
2. For video, adjust optional ST-GCN and CPU-performance sidebar settings.
3. Wait for the final dashboard and annotated video.
4. Optionally provide a Gemini API key and select **Generate coaching report**.
5. Select **Download coaching report PDF** to save the metrics, guard timeline,
   fatigue comparison, measured evidence, strengths, improvements, drills, and
   data limitations.

Only structured session metrics are sent to Gemini. The uploaded video, frames,
pose coordinates, API key, and generated PDF are not sent to Gemini or stored by
CornerCoach. The PDF is generated locally in memory and served through the
Streamlit download button.

## Models and decision rules

- `weights/yolo11s-pose.pt`: mandatory full-frame person tracking followed by
  cropped top-down pose extraction.
- `weights/stgcn/best_checkpoint.pt`: 11-frame ST-GCN inference.

The active ST-GCN checkpoint is loaded from `weights/stgcn/`. The retired LSTM
checkpoint is archived under `extras/retired_lstm/` and is not loaded by the
app. The YOLO pose checkpoint is unchanged.

A punch is counted only when the configured joint hand/punch confidence exceeds
the enforced minimum and the wrist-speed, outward-extension, peak-prominence,
and recovery gates also pass. Predictions below the threshold remain `IDLE`.

## Troubleshooting

- **PowerShell script execution is blocked:** use the full
  `powershell -ExecutionPolicy Bypass -File ...` commands shown above.
- **A checkpoint is missing:** restore the exact two files listed under
  Required production files and rerun setup.
- **Gemini report button is disabled:** pass `-GeminiApiKey` to `run.ps1`, set
  `GEMINI_API_KEY`, or place it in Streamlit secrets.
- **CPU inference is still slow:** keep detection at 320, pose at 480, batch
  size at 8, and live preview disabled. Test `CORNERCOACH_CPU_THREADS` values 2,
  4, and 8. If memory is limited, lower the batch size to 4. The two-stage
  detector-to-cropped-pose process remains mandatory.

Runtime logs are written under `logs/`. Training code and historical utilities
remain under `extras/` and are not needed to launch the app. The exact class
mapping, feature order, checkpoint hashes, and runtime interpretation are listed
in `weights/MODEL_FEATURE_GUIDE.md`.
