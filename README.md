readme_content = """
# YOLOv8 Smoke Detector for Coal Combustion

## Project Overview
This project aims to develop a computer vision model using YOLOv8 to detect smoke in images related to coal combustion. The primary goal is to identify and localize smoke, which can be critical for monitoring and safety in industrial settings.

## Execution Details

### 1. Environment Setup

First, the Google Drive was mounted to access the dataset, and the `ultralytics` library, which provides the YOLOv8 framework, was installed.

```python
from google.colab import drive
drive.mount('/content/drive')
!pip install ultralytics
```

### 2. Dataset Preparation

The dataset for training and validation is stored in Google Drive at `/content/drive/MyDrive/coal_combustion_image_dataset`. A `data.yaml` configuration file was created to inform YOLOv8 about the dataset's structure, including paths to training and validation images, and the single class 'smoke'.

```python
dataset_path = '/content/drive/MyDrive/coal_combustion_image_dataset'
data_yaml_content = f"""
path: {dataset_path}
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['smoke']
"""
with open('data.yaml', 'w') as f:
    f.write(data_yaml_content)
```

### 3. Model Training

A YOLOv8n (nano version) model, pre-trained on the COCO dataset, was loaded and fine-tuned on our custom smoke detection dataset for 25 epochs. The training process leveraged image augmentation and optimization techniques inherent to the YOLOv8 framework.

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data='data.yaml', epochs=25, imgsz=640)
```

### 4. Model Validation

After training, the model's performance was evaluated on the validation set. Key metrics were collected to assess its accuracy and robustness in detecting smoke.

**Validation Metrics:**
*   **Precision (P):** 0.8071
*   **Recall (R):** 0.5116
*   **mAP50:** 0.6882
*   **mAP50-95:** 0.3856
*   **F1 Score:** 0.6263

```python
metrics = model.val(data='data.yaml')
# Print metrics...
```

### 5. Model Export

The trained model was exported to ONNX format for potential deployment in various environments. The PyTorch state dictionary was also saved for further use within Python or for loading back into the `ultralytics` framework.

*   **ONNX Model Path:** `/content/runs/detect/train-2/weights/best.onnx`
*   **PyTorch State Dict Path:** `/content/yolov8n_smoke_detector.pt`

```python
export_path = model.export(format='onnx')
import torch
torch.save(model.state_dict(), 'yolov8n_smoke_detector.pt')
```

### 6. Model Testing (Inference)

To test the trained model, an image from the dataset's validation set was used. The model performs prediction on this image, and the results, including bounding boxes around detected smoke, are displayed.

```python
from ultralytics import YOLO
import os
from PIL import Image # Needed for displaying results

model_path = '/content/runs/detect/train-2/weights/best.pt'
model = YOLO(model_path)

dataset_path = '/content/drive/MyDrive/coal_combustion_image_dataset' # Ensure this is defined
validation_images_dir = os.path.join(dataset_path, 'valid', 'images')

if os.path.exists(validation_images_dir):
    all_images = [f for f in os.listdir(validation_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if all_images:
        test_image_name = all_images[0] # Using the first image for demonstration
        test_image_path = os.path.join(validation_images_dir, test_image_name)
        print(f"Using sample test image from validation set: {test_image_path}")
    else:
        test_image_path = "/path/to/your/test_image.jpg"
        print("No images found in validation set.")
else:
    test_image_path = "/path/to/your/test_image.jpg"
    print("Validation image directory not found.")

results = model.predict(source=test_image_path, conf=0.25) # conf sets confidence threshold

for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    display(Image.fromarray(im_array[..., ::-1])) # RGB conversion and display
```

## How to Run the Model and Get Outputs

To replicate the model's performance or use it for new predictions:

1.  **Mount Google Drive:** Ensure your Google Drive is mounted and the `coal_combustion_image_dataset` is accessible at the specified path.
2.  **Install Dependencies:** Run `!pip install ultralytics`.
3.  **Prepare `data.yaml`:** Create the `data.yaml` file as shown in section 2.
4.  **Load Trained Model:** Load the `best.pt` weights from the training output directory (e.g., `/content/runs/detect/train-2/weights/best.pt`). If you rerun training, this path might change.
5.  **Perform Inference:** Use `model.predict()` with your desired image path. Adjust the `conf` (confidence threshold) parameter as needed.
6.  **Visualize Results:** The `results` object from `model.predict()` can be used to plot bounding boxes directly onto the images. The `r.plot()` method returns a NumPy array suitable for display.

## Further Steps

*   **More Data:** Expand the dataset for more robust training.
*   **Hyperparameter Tuning:** Optimize training parameters (e.g., learning rate, batch size, augmentations) for better performance.
*   **Different YOLOv8 Models:** Experiment with other YOLOv8 variants (e.g., `yolov8m`, `yolov8l`) for different trade-offs between speed and accuracy.
*   **Deployment:** Integrate the exported ONNX model into a real-time application or edge device.
"""

with open('README.md', 'w') as f:
    f.write(readme_content)

print("README.md created successfully in the /content directory.")