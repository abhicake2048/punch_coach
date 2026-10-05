# Archived runtime files

This directory is not imported by the CornerCoach production application.

- `retired_lstm/production_weight/best_checkpoint.pt` is the LSTM checkpoint
  removed when production inference was converted to ST-GCN-only operation.

The only active classifier checkpoint is now
`../weights/stgcn/best_checkpoint.pt`. The active YOLO pose checkpoint remains
`../weights/yolo11s-pose.pt`.
