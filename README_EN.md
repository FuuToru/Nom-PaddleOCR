# Nom Character Recognition Project with PaddleOCR

**English | [Tiếng Việt](README.md)**

## Introduction

This project uses the PaddleOCR framework to build an Optical Character Recognition (OCR) system for Nom characters. The project consists of two main components:

1. **Text Detection**: Detecting and localizing text regions in images
2. **Text Recognition**: Recognizing text content from detected regions

## Project Structure

```
├── src/
│   └── paddle-ocr.ipynb    # Main notebook containing the entire pipeline
├── data/
│   └── nomna.txt          # Link to Kaggle dataset
├── slide/
│   └── Slide.pdf          # Presentation document
└── README.md              # This guide file
```

## Dataset

### Data Source
- **Dataset**: NomNa OCR Dataset
- **Kaggle Link**: https://www.kaggle.com/datasets/quandang/nomnaocr
- **Content**: Dataset containing images of ancient book pages with Nom characters and corresponding labels

### Data Structure
- `Pages/`: Contains original book page images
- `Patches/`: Contains patches cropped from original images
- `All.txt`: File containing all training labels
- `Validate.txt`: File containing validation labels
- `PaddleOCR-Train.txt`: Training file formatted for PaddleOCR
- `PaddleOCR-Validate.txt`: Validation file formatted for PaddleOCR

## Installation and Environment

### System Requirements
- Python 3.7+
- CUDA (recommended for training)
- GPU with at least 8GB VRAM

### Installing Dependencies

```bash
# Install PaddlePaddle
python -m pip install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Clone PaddleOCR repository
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR

# Install requirements
pip install -r requirements.txt
```

### Data Preparation

1. Download dataset from Kaggle:
```bash
# Use Kaggle API or download directly from link
kaggle datasets download -d quandang/nomnaocr
```

2. Extract and place in `data/` directory

## Training

### 1. Training Text Detection (DBNet)

```bash
# Create config file for detection
python -c "
det_yaml_config = '''
Global:
    use_gpu: true
    epoch_num: 15
    log_smooth_window: 20
    print_batch_step: 369
    save_model_dir: ./output/db_nomna/
    save_epoch_step: 5
    eval_batch_step: [0, 369]
    cal_metric_during_train: False
    pretrained_model: ./pretrain_models/MobileNetV3_large_x0_5_pretrained
    checkpoints:
    save_inference_dir:
    use_visualdl: False
    infer_img: doc/imgs_en/img_10.jpg
    save_res_path: ./output/det_db/predicts_db.txt

Architecture:
    model_type: det
    algorithm: DB
    Transform:
    Backbone:
        name: MobileNetV3
        scale: 0.5
        model_name: large
        disable_se: True
    Neck:
        name: DBFPN
        out_channels: 256
    Head:
        name: DBHead
        k: 50

Loss:
    name: DBLoss
    balance_loss: true
    main_loss_type: DiceLoss
    alpha: 5
    beta: 10
    ohem_ratio: 3

Optimizer:
    name: Adam
    beta1: 0.9
    beta2: 0.999
    lr:
        name: Cosine
        learning_rate: 0.001
        warmup_epoch: 2
    regularizer:
        name: 'L2'
        factor: 0

PostProcess:
    name: DBPostProcess
    thresh: 0.3
    box_thresh: 0.6
    max_candidates: 1000
    unclip_ratio: 1.5

Metric:
    name: DetMetric
    main_indicator: hmean

Train:
    dataset:
        name: SimpleDataSet
        data_dir: /path/to/dataset/
        label_file_list:
            - /path/to/PaddleOCR-Train.txt
        ratio_list: [1.0]
        transforms:
            - DecodeImage:
                  img_mode: BGR
                  channel_first: False
            - DetLabelEncode:
            - IaaAugment:
                  augmenter_args:
                      - { 'type': Affine, 'args': { 'rotate': [-10, 10] } }
                      - { 'type': Resize, 'args': { 'size': [0.5, 3] } }
            - EastRandomCropData:
                  size: [640, 640]
                  max_tries: 50
                  keep_ratio: true
            - MakeBorderMap:
                  shrink_ratio: 0.4
                  thresh_min: 0.3
                  thresh_max: 0.7
            - MakeShrinkMap:
                  shrink_ratio: 0.4
                  min_text_size: 8
            - NormalizeImage:
                  scale: 1./255.
                  mean: [0.485, 0.456, 0.406]
                  std: [0.229, 0.224, 0.225]
                  order: 'hwc'
            - ToCHWImage:
            - KeepKeys:
                  keep_keys: ['image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask']
    loader:
        shuffle: True
        drop_last: False
        batch_size_per_card: 8
        num_workers: 0

Eval:
    dataset:
        name: SimpleDataSet
        data_dir: /path/to/dataset/
        label_file_list:
            - /path/to/PaddleOCR-Validate.txt
        transforms:
            - DecodeImage:
                  img_mode: BGR
                  channel_first: False
            - DetLabelEncode:
            - DetResizeForTest:
            - NormalizeImage:
                  scale: 1./255.
                  mean: [0.485, 0.456, 0.406]
                  std: [0.229, 0.224, 0.225]
                  order: 'hwc'
            - ToCHWImage:
            - KeepKeys:
                  keep_keys: ['image', 'shape', 'polys', 'ignore_tags']
    loader:
        shuffle: False
        drop_last: False
        batch_size_per_card: 1
        num_workers: 0
'''
with open('dbnet_train.yml', 'w') as f:
    f.write(det_yaml_config)
"

# Run detection training
python -m paddle.distributed.launch --gpus '0,1' tools/train.py -c dbnet_train.yml
```

### 2. Training Text Recognition (SAR)

```bash
# Create config file for recognition
python -c "
sar_yaml_config = '''
Global:
  use_gpu: true
  epoch_num: 50
  log_smooth_window: 20
  print_batch_step: 20
  save_model_dir: ./sar_rec_nomna/
  save_epoch_step: 10
  eval_batch_step: [0, 2000]
  cal_metric_during_train: True
  pretrained_model:
  checkpoints:
  save_inference_dir:
  use_visualdl: False
  infer_img: doc/imgs_words/ch/word_1.jpg
  character_dict_path: /path/to/nomna_dict.txt
  character_type: ch
  max_text_length: 25
  infer_mode: False
  use_space_char: False
  save_res_path: ./output/rec/predicts_sar.txt

Architecture:
  model_type: rec
  algorithm: SAR
  Transform:
  Backbone:
    name: ResNet31
    layers: 31
  Head:
    name: SARHead
    enc_dim: 512
    max_text_length: 25

Loss:
  name: SARLoss

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001
    warmup_epoch: 5
  regularizer:
    name: 'L2'
    factor: 0.00001

PostProcess:
  name: SARLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: /path/to/rotated_images/
    label_file_list:
      - /path/to/rec_gt_train.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - SARLabelEncode:
      - SARRecResizeImg:
          image_shape: [3, 48, 168]
          width_downsample_ratio: 0.25
      - KeepKeys:
          keep_keys: ['image', 'label']
  loader:
    shuffle: True
    batch_size_per_card: 64
    drop_last: True
    num_workers: 8
    use_shared_memory: False

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: /path/to/rotated_images/
    label_file_list:
      - /path/to/rec_gt_val.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - SARLabelEncode:
      - SARRecResizeImg:
          image_shape: [3, 48, 168]
          width_downsample_ratio: 0.25
      - KeepKeys:
          keep_keys: ['image', 'label']
  loader:
    shuffle: False
    drop_last: False
    batch_size_per_card: 64
    num_workers: 4
    use_shared_memory: False
'''
with open('rec_sar_train.yml', 'w') as f:
    f.write(sar_yaml_config)
"

# Run recognition training
python -m paddle.distributed.launch --gpus '0,1' tools/train.py -c rec_sar_train.yml
```

## Evaluation

### Evaluating Detection Model

```bash
# Create evaluation config for detection
python tools/eval.py -c dbnet_eval.yml
```

### Evaluating Recognition Model

```bash
# Create evaluation config for recognition
python tools/eval.py -c rec_sar_eval.yml
```

## Inference

### Using Complete Pipeline

```python
import os
import cv2
import json
import subprocess
from pathlib import Path

def run_detection(img_path, det_config, save_dir):
    """Run detection and return result file"""
    os.makedirs(save_dir, exist_ok=True)
    cmd = f"""python3 tools/infer_det.py \
        -c {det_config} \
        -o Global.infer_img="{img_path}" \
        Global.save_res_path="{save_dir}/det_result.txt" \
        Global.save_crop_res=False"""
    subprocess.run(cmd, shell=True, check=True)
    return os.path.join(save_dir, "det_result.txt")

def crop_and_rotate(det_file, crop_dir, rotate_dir, src_img):
    """Crop and rotate images based on detection results"""
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(rotate_dir, exist_ok=True)
    
    # Read detection results
    with open(det_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            image_file, json_str = line.split('\t', 1)
            if image_file != src_img:
                continue
                
            data = json.loads(json_str)
            img = cv2.imread(src_img)
            
            # Process each detected text region
            for i, item in enumerate(data):
                pts = item["points"]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                
                # Crop image
                crop_img = img[y_min:y_max, x_min:x_max]
                crop_path = os.path.join(crop_dir, f"{i:03}.jpg")
                cv2.imwrite(crop_path, crop_img)
                
                # Rotate image 90 degrees counterclockwise
                rotated_img = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotate_path = os.path.join(rotate_dir, f"{i:03}.jpg")
                cv2.imwrite(rotate_path, rotated_img)
    
    return [os.path.join(rotate_dir, f) for f in sorted(os.listdir(rotate_dir))]

def run_recognition_per_image(img_path, rec_config):
    """Run recognition on a single image"""
    cmd = (
        f"python3 tools/infer_rec.py -c {rec_config} "
        f"-o Global.infer_img=\"{img_path}\" "
        f"Global.use_space_char=False"
    )
    output = subprocess.check_output(cmd, shell=True, text=True)
    
    # Parse results
    for line in output.splitlines():
        if "result:" in line:
            return line.split("result:")[1].strip().strip("[]'\"")
    return ""

# Using pipeline
def inference_pipeline(image_path, det_config, rec_config, output_dir):
    """Complete inference pipeline"""
    
    # 1. Detection
    det_file = run_detection(image_path, det_config, output_dir)
    
    # 2. Crop and rotate
    crop_dir = os.path.join(output_dir, "crops")
    rotate_dir = os.path.join(output_dir, "rotated")
    rotated_images = crop_and_rotate(det_file, crop_dir, rotate_dir, image_path)
    
    # 3. Recognition
    results = []
    for img_path in rotated_images:
        text = run_recognition_per_image(img_path, rec_config)
        results.append(text)
    
    # 4. Combine results
    final_text = ''.join(results)
    
    print(f"Recognition result: {final_text}")
    return final_text

# Usage example
if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    det_config = "dbnet_eval.yml"
    rec_config = "rec_sar_eval.yml"
    output_dir = "./inference_output"
    
    result = inference_pipeline(image_path, det_config, rec_config, output_dir)
```

## Performance Evaluation

The project uses the following metrics for evaluation:

- **Detection**: Precision, Recall, F1-score based on IoU
- **Recognition**: Sequence accuracy

## Results

Detailed performance results are presented in the file `slide/Slide.pdf`.

## References

- [PaddleOCR Official Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [NomNa OCR Dataset](https://www.kaggle.com/datasets/quandang/nomnaocr)
- [DBNet Paper](https://arxiv.org/abs/1911.08947)
- [SAR Paper](https://arxiv.org/abs/1811.00751)

## Contact

If you have any questions about the project, please create an issue or contact via email huutri231103@gmail.com. 