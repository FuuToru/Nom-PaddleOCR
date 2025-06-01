# Project Nhận Dạng Chữ Nôm với PaddleOCR

**[English Version](README_EN.md) | Tiếng Việt**

## Giới thiệu

Đây là project sử dụng framework PaddleOCR để xây dựng hệ thống nhận dạng ký tự chữ Nôm (Optical Character Recognition - OCR). Project bao gồm hai thành phần chính:

1. **Text Detection**: Phát hiện và định vị vùng chứa văn bản trong ảnh
2. **Text Recognition**: Nhận dạng nội dung văn bản từ các vùng đã được phát hiện

## Cấu trúc Project

```
├── src/
│   └── paddle-ocr.ipynb    # Notebook chính chứa toàn bộ pipeline
├── data/
│   └── nomna.txt          # Link đến dataset Kaggle
├── slide/
│   └── Slide.pdf          # Tài liệu thuyết trình
└── README.md              # File hướng dẫn này
```

## Dataset

### Nguồn dữ liệu
- **Dataset**: NomNa OCR Dataset
- **Link Kaggle**: https://www.kaggle.com/datasets/quandang/nomnaocr
- **Nội dung**: Bộ dữ liệu chứa ảnh các trang sách cổ với chữ Nôm và nhãn tương ứng

### Cấu trúc dữ liệu
- `Pages/`: Chứa ảnh các trang sách gốc
- `Patches/`: Chứa các patch đã được cắt từ ảnh gốc
- `All.txt`: File chứa tất cả nhãn training
- `Validate.txt`: File chứa nhãn validation
- `PaddleOCR-Train.txt`: File training đã được format cho PaddleOCR
- `PaddleOCR-Validate.txt`: File validation đã được format cho PaddleOCR

## Cài đặt và Môi trường

### Yêu cầu hệ thống
- Python 3.7+
- CUDA (khuyến nghị cho training)
- GPU với ít nhất 8GB VRAM

### Cài đặt dependencies

```bash
# Cài đặt PaddlePaddle
python -m pip install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Clone PaddleOCR repository
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR

# Cài đặt requirements
pip install -r requirements.txt
```

### Chuẩn bị dữ liệu

1. Tải dataset từ Kaggle:
```bash
# Sử dụng Kaggle API hoặc tải trực tiếp từ link
kaggle datasets download -d quandang/nomnaocr
```

2. Giải nén và đặt vào thư mục `data/`

## Training

### 1. Training Text Detection (DBNet)

```bash
# Tạo file config cho detection
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

# Chạy training detection
python -m paddle.distributed.launch --gpus '0,1' tools/train.py -c dbnet_train.yml
```

### 2. Training Text Recognition (SAR)

```bash
# Tạo file config cho recognition
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

# Chạy training recognition
python -m paddle.distributed.launch --gpus '0,1' tools/train.py -c rec_sar_train.yml
```

## Evaluation

### Đánh giá Detection Model

```bash
# Tạo config evaluation cho detection
python tools/eval.py -c dbnet_eval.yml
```

### Đánh giá Recognition Model

```bash
# Tạo config evaluation cho recognition
python tools/eval.py -c rec_sar_eval.yml
```

## Inference

### Sử dụng Pipeline Hoàn chỉnh

```python
import os
import cv2
import json
import subprocess
from pathlib import Path

def run_detection(img_path, det_config, save_dir):
    """Chạy detection và trả về file kết quả"""
    os.makedirs(save_dir, exist_ok=True)
    cmd = f"""python3 tools/infer_det.py \
        -c {det_config} \
        -o Global.infer_img="{img_path}" \
        Global.save_res_path="{save_dir}/det_result.txt" \
        Global.save_crop_res=False"""
    subprocess.run(cmd, shell=True, check=True)
    return os.path.join(save_dir, "det_result.txt")

def crop_and_rotate(det_file, crop_dir, rotate_dir, src_img):
    """Cắt và xoay ảnh dựa trên kết quả detection"""
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(rotate_dir, exist_ok=True)
    
    # Đọc kết quả detection
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
            
            # Xử lý từng vùng text được detect
            for i, item in enumerate(data):
                pts = item["points"]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                
                # Cắt ảnh
                crop_img = img[y_min:y_max, x_min:x_max]
                crop_path = os.path.join(crop_dir, f"{i:03}.jpg")
                cv2.imwrite(crop_path, crop_img)
                
                # Xoay ảnh 90 độ ngược chiều kim đồng hồ
                rotated_img = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotate_path = os.path.join(rotate_dir, f"{i:03}.jpg")
                cv2.imwrite(rotate_path, rotated_img)
    
    return [os.path.join(rotate_dir, f) for f in sorted(os.listdir(rotate_dir))]

def run_recognition_per_image(img_path, rec_config):
    """Chạy recognition cho một ảnh"""
    cmd = (
        f"python3 tools/infer_rec.py -c {rec_config} "
        f"-o Global.infer_img=\"{img_path}\" "
        f"Global.use_space_char=False"
    )
    output = subprocess.check_output(cmd, shell=True, text=True)
    
    # Parse kết quả
    for line in output.splitlines():
        if "result:" in line:
            return line.split("result:")[1].strip().strip("[]'\"")
    return ""

# Sử dụng pipeline
def inference_pipeline(image_path, det_config, rec_config, output_dir):
    """Pipeline inference hoàn chỉnh"""
    
    # 1. Detection
    det_file = run_detection(image_path, det_config, output_dir)
    
    # 2. Crop và rotate
    crop_dir = os.path.join(output_dir, "crops")
    rotate_dir = os.path.join(output_dir, "rotated")
    rotated_images = crop_and_rotate(det_file, crop_dir, rotate_dir, image_path)
    
    # 3. Recognition
    results = []
    for img_path in rotated_images:
        text = run_recognition_per_image(img_path, rec_config)
        results.append(text)
    
    # 4. Kết hợp kết quả
    final_text = ''.join(results)
    
    print(f"Kết quả nhận dạng: {final_text}")
    return final_text

# Ví dụ sử dụng
if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    det_config = "dbnet_eval.yml"
    rec_config = "rec_sar_eval.yml"
    output_dir = "./inference_output"
    
    result = inference_pipeline(image_path, det_config, rec_config, output_dir)
```


## Đánh giá Hiệu suất

Project sử dụng các metrics sau để đánh giá:

- **Detection**: Precision, Recall, F1-score dựa trên IoU
- **Recognition**: Sequence accuracy

## Kết quả

Các kết quả chi tiết về hiệu suất model được trình bày trong file `slide/Slide.pdf`.

## Tài liệu tham khảo

- [PaddleOCR Official Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [NomNa OCR Dataset](https://www.kaggle.com/datasets/quandang/nomnaocr)
- [DBNet Paper](https://arxiv.org/abs/1911.08947)
- [SAR Paper](https://arxiv.org/abs/1811.00751)

## Liên hệ

Nếu có thắc mắc về project, vui lòng tạo issue hoặc liên hệ qua email huutri231103@gmail.com. 