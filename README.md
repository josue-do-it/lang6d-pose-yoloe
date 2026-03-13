# Open Vocabulary 6D Object Pose Estimation using YOLOE
Research project exploring open-vocabulary object detection and 6D pose estimation by combining YOLOE with model-free pose estimation methods for robotic manipulation of unseen objects.


# Proposal 1

## Title
**Open-vocabulary object pose estimation using YOLOE**  
*Josué Romaric Edou*

---

## Theme
Computer Vision

---

## Prerequisites

- Python
- PyTorch
- Basic Math: Linear algebra (transformation matrices) and deep learning fundamentals
- Asset: Experience with CUDA and OpenCV

---

## Project Description

In this project, the student will integrate a state-of-the-art object detection method with a pose estimation framework to determine the **6D pose of novel objects**.

The proposed detection method is **YOLOE**, a real-time open-vocabulary detector that identifies objects based on vision and simple text prompts. The YOLOE model is publicly available and ready to be used as is.

The primary technical challenge is using YOLOE in a **"frozen" (off-the-shelf)** manner to drive a downstream **6D pose estimator**.

The pipeline will work as follows:

1. **YOLOE detection**
   - YOLOE detects the precise **2D location** and **segmentation** of an object based on a user's text prompt.

2. **Pose estimation**
   - This 2D output will serve as the input for a **model-free pose estimation method** (such as Any6D).

3. **6D pose regression**

The pose estimator will regress the object’s **6D pose**:


This capability is critical in **robotics for the accurate manipulation of unseen objects**.

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

## Useful Links

### YOLOE

YOLOE Paper  
https://arxiv.org/pdf/2503.07465

YOLOE Implementation (recommended)  
https://docs.ultralytics.com/models/yoloe/

YOLOE Original Repository  
https://github.com/THU-MIG/yoloe

---

### Any6D

Any6D Website (paper and code)  
https://sites.google.com/view/taeyeop-lee/any6d

---

### Review Papers (Object Pose Estimation)

Review Paper 1  
https://arxiv.org/pdf/2405.07801

Review Paper 2  
https://www.mdpi.com/1424-8220/24/4/1076
