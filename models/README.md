# MediaPipe 模型（Photo-Stitch）

首次运行会按需下载到本目录。也可手动：

## Face Landmarker（脸窗 / 脸罩）

`face_landmarker.task`

```powershell
curl.exe -L -o models\face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

失败回退 OpenCV Haar + 下巴精炼。

## Hand Landmarker（人物+动作：手罩）

`hand_landmarker.task`

```powershell
curl.exe -L -o models\hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

用于脸外手足密度场；失败回退肤色 Lab 近邻。

## Pose Landmarker Lite（臂/肩引导）

`pose_landmarker_lite.task`

```powershell
curl.exe -L -o models\pose_landmarker_lite.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
```

肩-肘-腕粗带作袖臂引导；道具暗块优先在手部先验附近。失败不影响出图。
