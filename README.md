# Open Vocabulary 6D Object Pose Estimation using YOLOE

Research project exploring open-vocabulary object detection and 6D pose estimation by combining YOLOE with model-free pose estimation methods for robotic manipulation of unseen objects.
## Tools
- Python
- PyTorch
- Basic Math: Linear algebra (transformation matrices) and deep learning fundamentals
- Asset: Experience with CUDA and OpenCV
---

## Project Description

In this project, we  will integrate a state-of-the-art object detection method with a pose estimation framework to determine the **6D pose of novel objects**. The proposed detection method is **YOLOE**, a real-time open-vocabulary detector that identifies objects based on vision and simple text prompts. The YOLOE model is publicly available and ready to be used as is.The primary technical challenge is using YOLOE in a **"frozen" (off-the-shelf)** manner to drive a downstream **6D pose estimator**.

The pipeline will work as follows:

1. **YOLOE detection**
   - YOLOE detects the precise **2D location** and **segmentation** of an object based on a user's text prompt.

2. **Pose estimation**
   - This 2D output will serve as the input for a **model-free pose estimation method** (such as Any6D).

3. **6D pose regression**
The pose estimator will regress the object’s 6D pose: This capability is critical in robotics for the accurate manipulation of unseen objects.
---

## Project Phases
The project includes the following phases:

### 1. Understanding key concepts
- Pose estimation in computer vision
- Basic math behind pose estimation
- Matrix conversions
- How the pre-trained YOLOE model works

### 2. Selecting a pose estimation method
- Example: **Any6D**
- Study and understand its implementation and code

### 3. Building the pipeline
- Connect the **YOLOE detection output** to the **pose estimation pipeline**
- Produce a **6D pose result**

The pipeline will be validated by:
- Providing an **image input**
- Running the model
- **Verifying the predicted 6D pose**
---
# Reading Schedule

| Date | Week | Paper | Link | Status | Notes |
|------|------|-------|------|--------|-------|
| 14-03-2026 | Week 1 | [Deep Learning-Based Object Pose Estimation: A Comprehensive Survey](https://arxiv.org/pdf/2405.07801) | PDF | ⬜ Not started | [Notes](notes/pose_estimation_review_2024.md) |
| 16-03-2026 | Week 1 | [A Survey of 6DoF Object Pose Estimation Methods for Different Application Scenarios](https://www.mdpi.com/1424-8220/24/4/1076) | PDF | ⬜ Not started | [Notes](notes/pose_estimation_review_mdpi.md) |
| 18-03-2026 | Week 2 | [YOLOE: Real-Time Open-Vocabulary Object Detection](https://arxiv.org/pdf/2503.07465) | PDF | ⬜ Not started | [Notes](notes/yoloe_paper.md) |
| 20-03-2026 | Week 2 | [Ultralytics YOLO Docs](https://docs.ultralytics.com/models/yoloe/) | Web | ⬜ Not started | [Notes](notes/yoloe_documentation.md) |
| 22-03-2026 | Week 3 | [Any6D: Model-free 6D Pose Estimation of Novel Objects CVPR 2025](https://sites.google.com/view/taeyeop-lee/any6d) | Web | ⬜ Not started | [Notes](notes/any6d_paper.md) |
| 24-03-2026 | Week 3 | [Any6D ](https://github.com/taeyeopl/any6d) | GitHub | ⬜ Not started | [Notes](notes/any6d_code.md) |
