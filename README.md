# CornerCoach production inference

CornerCoach analyzes an uploaded boxing video on CPU with YOLO11 pose and a
selectable trained LSTM or ST-GCN. It reports punch counts, hand and punch type,
guard discipline, fatigue indicators, a metric-grounded Gemini coaching report,
an annotated video, and a downloadable PDF report.

## Start on Windows in three steps

Requirements: Windows 10/11, 64-bit Python 3.10-3.12, and the three weight files
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
weights/lstm/best_checkpoint.pt
weights/stgcn/best_checkpoint.pt
requirements.txt
setup.ps1 and run.ps1
setup.sh and run.sh
```

The setup scripts stop with a clear error if a required checkpoint is missing.

## Recommended CPU settings

The app defaults to the faster configuration:

- `Fast single-pass YOLO pose`: enabled. YOLO11s performs tracking and pose in
  one inference instead of running the pose network a second time on the crop.
- `YOLO input size`: 480 pixels.
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
faster. If the tracked boxer is small or the single-pass pose quality is
insufficient, disable `Fast single-pass YOLO pose` in the sidebar. That restores
the original two-stage detector -> padded crop -> pose path, but it is expected
to be substantially slower.

The application does not skip inference frames because the trained checkpoints
use 11-frame motion sequences and fast punches may last only 5-8 frames.

## Using the app

1. Upload an MP4/MOV video or JPG/PNG image.
2. For video, choose LSTM or ST-GCN and adjust optional sidebar settings.
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

- `weights/yolo11s-pose.pt`: single-pass person tracking and pose, or optional
  two-stage tracking plus cropped pose.
- `weights/lstm/best_checkpoint.pt`: 11-frame LSTM inference.
- `weights/stgcn/best_checkpoint.pt`: 11-frame ST-GCN inference.

A punch is counted only when the configured joint hand/punch confidence exceeds
the enforced minimum and the wrist-speed, outward-extension, peak-prominence,
and recovery gates also pass. Predictions below the threshold remain `IDLE`.

## Troubleshooting

- **PowerShell script execution is blocked:** use the full
  `powershell -ExecutionPolicy Bypass -File ...` commands shown above.
- **A checkpoint is missing:** restore the exact three files listed under
  Required production files and rerun setup.
- **Gemini report button is disabled:** pass `-GeminiApiKey` to `run.ps1`, set
  `GEMINI_API_KEY`, or place it in Streamlit secrets.
- **CPU inference is still slow:** keep 480 pixels, single-pass pose enabled, and
  live preview disabled. Test `CORNERCOACH_CPU_THREADS` values 2, 4, and 8 on the
  target machine.
- **Pose quality drops in fast mode:** turn off single-pass pose for that video.
  This is a quality/speed tradeoff; it does not change the trained classifier.

Runtime logs are written under `logs/`. Training code and historical utilities
remain under `Extras/` and are not needed to launch the app.
