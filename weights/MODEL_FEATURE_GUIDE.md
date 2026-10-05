# CornerCoach model feature and output guide

This guide describes the contract embedded in the active ST-GCN checkpoint.
The runtime reads this contract directly from the checkpoint and validates it
before loading model weights.

## Outputs

The model uses two classification heads:

- Hand head: `none`, `left`, `right`
- Punch-type head: `background`, `cross`, `jab`, `hook`, `uppercut`

The external nine-action interpretation is:

1. `IDLE` = hand `none` and punch type `background`
2. `Left Cross`
3. `Left Jab`
4. `Left Hook`
5. `Left Uppercut`
6. `Right Cross`
7. `Right Jab`
8. `Right Hook`
9. `Right Uppercut`

A punch is emitted only when both heads select a positive combination and the
joint confidence, wrist speed, outward extension, peak prominence, and recovery
gates all pass. A `none` hand, `background` punch type, incomplete pose window,
or sub-threshold prediction is displayed as `IDLE` and is not counted.

## Shared temporal contract

- Feature schema version: `1`
- Window length: `11` consecutive frames
- Pose source: COCO-17 keypoints from the tracked, padded boxer crop
- Coordinate preparation: crop keypoints are mapped to the original frame and
  normalized at the neck before kinematics are calculated
- Normalization: each checkpoint's saved training-only mean and standard
  deviation are applied in the exact saved feature order

## ST-GCN inputs

For each of the 17 COCO joints, the graph branch consumes these seven channels
in exact order:

1. `x`
2. `y`
3. `confidence`
4. `vx`
5. `vy`
6. `ax`
7. `ay`

Its parallel kinematic branch consumes these 31 features in exact order:

1. `torso_orientation_deg`
2. `left_elbow_angle`
3. `left_elbow_angular_velocity`
4. `left_wrist_neck_x`
5. `left_wrist_neck_y`
6. `left_wrist_shoulder_x`
7. `left_wrist_shoulder_y`
8. `left_wrist_reach`
9. `left_wrist_extension_velocity`
10. `left_wrist_extension_acceleration`
11. `left_wrist_velocity_x`
12. `left_wrist_velocity_y`
13. `left_wrist_speed`
14. `left_wrist_acceleration_x`
15. `left_wrist_acceleration_y`
16. `left_wrist_acceleration_magnitude`
17. `right_elbow_angle`
18. `right_elbow_angular_velocity`
19. `right_wrist_neck_x`
20. `right_wrist_neck_y`
21. `right_wrist_shoulder_x`
22. `right_wrist_shoulder_y`
23. `right_wrist_reach`
24. `right_wrist_extension_velocity`
25. `right_wrist_extension_acceleration`
26. `right_wrist_velocity_x`
27. `right_wrist_velocity_y`
28. `right_wrist_speed`
29. `right_wrist_acceleration_x`
30. `right_wrist_acceleration_y`
31. `right_wrist_acceleration_magnitude`

## Active files and verified hashes

- `stgcn/best_checkpoint.pt`  
  SHA-256: `DE19349C99F9591006C14393622D7632510CFD67A15BD78116EF556B42B6BAE3`
- `yolo11s-pose.pt` supplies person detection/tracking and pose keypoints.

Only ST-GCN is loaded by production inference. The retired LSTM checkpoint is
kept under `extras/retired_lstm/` for rollback and is not referenced by the app.
