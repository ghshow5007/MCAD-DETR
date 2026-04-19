# MCAD-DETR: A Real-Time Lightweight Tobacco Leaf Disease and Damage Detection Model

## Abstract

Real-time and accurate detection of tobacco leaf diseases and damage is essential for agricultural modernization. However, complex field backgrounds, highly similar pathological characteristics, and the inherent trade-off between high precision and lightweight design remain significant challenges. To address these issues, we propose **MCAD-DETR** (**M**ulti-scale **C**omplementary **A**dditive **D**iffusion **De**tection **Tr**ansformer), a lightweight real-time detection model. First, we construct an **Efficient Feature Complementary Mapping Network (EFCMNet)** as the feature extraction network, which enhances feature extraction capabilities while maintaining a low parameter count. We also introduce the **Efficient Additive Attention Block (EAA Block)** to decouple high- and low-frequency features, thereby strengthening the representation of pathological details and suppressing background interference. Additionally, a **Tri-Focal Diffusion Feature Pyramid Network (TFDFPN)** is designed to improve multi-scale detection performance and optimize model efficiency. To address class imbalance and small object detection in the dataset, we propose a multi-dimensional joint loss function. Experimental results demonstrate that MCAD-DETR achieves an \( mAP_{50} \) of 89.3% and a Recall of 83.0%, outperforming the RT-DETR baseline by 2.0% and 1.5%, respectively. The model parameters are reduced from 19.9 to 6.48M, the storage size decreases from 77.0 to 25.3MB, and the computational cost is lowered from 57.0 to 46.1 GFLOPs, representing reductions of 67.44%, 67.14%, and 19.12%, respectively. Compared with existing methods, MCAD-DETR provides superior detection accuracy with substantially lower complexity, making it well-suited for real-time tobacco disease and damage detection applications.

![Structure of MCAD-DETR](images/MCAD-DETR.jpg)

## Performance

The following table shows the performance comparison between the baseline RT-DETR and our proposed MCAD-DETR, based on the ablation study results (Table 7). MCAD-DETR achieves higher precision and recall while significantly reducing the computational overhead and model size.

| Model | Precision (%) | Recall (%) | F1-score (%) | mAP50 (%) | mAP75 (%) | mAP50:95 (%) | GFLOPs | FPS | Params (M) | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RT-DETR (Baseline)** | 90.5 | 81.5 | 85.7 | 87.3 | 73.8 | 69.0 | 57.0 | **91.6** | 19.9 | 77.0 |
| **MCAD-DETR (Ours)** | **91.2** | **83.0** | **86.8** | **89.3** | **75.5** | **70.3** | **46.1** | 78.9 | **6.48** | **25.3** |

### Key Improvements:
- **Accuracy:** **mAP50** increased by **2.0%** and **Recall** improved by **1.5%**.
- **Efficiency:** **Parameters** reduced by **67.44%** (from 19.9M to 6.48M).
- **Lightweight:** **Storage Size** decreased by **67.14%** (from 77.0MB to 25.3MB).
- **Computation:** **GFLOPs** lowered by **19.12%** (from 57.0 to 46.1).

## References & Credits

Our work is built upon and inspired by the RT-DETR architecture. For more information regarding the baseline model and its implementation, please refer to:

- **Ultralytics RT-DETR Implementation:** [Ultralytics GitHub - Real-time DETR](https://github.com/ultralytics/ultralytics)
- **Original RT-DETR Paper:** [DETRs Beat YOLOs on Real-time Object Detection (CVPR 2024)](https://arxiv.org/abs/2304.08069)

## Key Components

### 1. EFCMNet (Backbone)
The backbone is built upon **Feature Complementary Mapping (FCM)** blocks and **Multi-Scale Decomposed Kernel (MSDK)** blocks. 
- **FCM Blocks**: Employ a differentiated channel partition strategy and dual-branch architecture (spatial and channel attention) to decouple high- and low-frequency features, suppressing background noise.
- **MSDK Block**: Positioned at the $P_4$ stage, it uses parallel depthwise convolution branches with $3 \times 3$, $5 \times 5$, and $7 \times 7$ receptive fields. Large kernels are decomposed into 1D asymmetric convolutions to capture irregular lesion edges with low computational overhead.

### 2. Efficient Additive Attention (EAA) Block
Replacing the original AIFI in RT-DETR, the EAA Block achieves global context modeling through linear element-wise interactions instead of quadratic-complexity matrix multiplications. It effectively integrates low-frequency global semantics with high-frequency spatial details (edges and textures).

### 3. TFDFPN (Neck)
The **Tri-Focal Diffusion Feature Pyramid Network** optimizes feature fusion by:
- **Discarding the $P_5$ layer**: Preventing the loss of spatial details for tiny targets during excessive downsampling.
- **Three-Scale Feature Aggregation (TSFA) Module**: Aggregating features from $P_2$, $P_3$, and $S_4$ (enhanced by EAA) to achieve deep cross-scale complementarity.
- **Fast Normalized Fusion (FN Fusion)**: Using learnable weights to adaptively balance information from different scales.

## Project Structure

```text
MCAD-DETR/
├── README.md                # Project documentation
├── dataset/                 # Dataset utilities and placeholder
│   ├── images/              # (Empty) Download from Baidu Netdisk
│   ├── labels/              # (Empty) Download from Baidu Netdisk
│   ├── data.yaml            # Dataset configuration for training
│   ├── split_data.py        # Dataset splitting utility (Train/Val/Test)
│   ├── xml2txt.py          # XML to YOLO format converter
│   └── yolo2coco.py        # YOLO to COCO format converter
├── images/                  # Figures for README documentation
│   ├── dataset.jpg          # Dataset samples visualization
│   └── MCAD-DETR.jpg        # Model architecture diagram
└── ultralytics/             # Core implementation based on Ultralytics
    ├── cfg/                 # Configuration files
    │   ├── default.yaml     # Default training hyperparameters
    │   └── models/
    │       └── rt-detr/
    │           ├── MCAD-DETR.yaml    # Proposed MCAD-DETR architecture
    │           └── rtdetr-r18.yaml   # Baseline RT-DETR-ResNet18
    ├── nn/                  # Neural network modules
    │   └── modules/
    │       ├── attention.py   # Implementation of EAA Block
    │       ├── block.py       # Implementation of FCM, MSDK, TSFA, and Fusion
    │       ├── conv.py        # Custom convolution modules
    │       ├── head.py        # RTDETRDecoder and prediction heads
    │       └── transformer.py # Transformer-related components
    └── utils/
        └── loss.py          # Implementation of SVFL and Focaler_MPDIoU
```

## Dataset

The tobacco leaf disease and damage dataset used in this study consists of **1,560 images** (640×640 resolution) with **9,534 annotations**. The data is split into training, validation, and test sets with a ratio of approximately **7:1:2**.

![Tobacco Leaf Disease and Damage Dataset](images/dataset.jpg)

### Download Link

The complete data repository, including the proposed **tobacco leaf disease and damage dataset**, the comparative datasets (**PlantDoc**, **MCD**, and **SDNET 2025**), as well as the images from **challenging scenarios** used for robustness analysis, has been uploaded to Baidu Netdisk:

- **Link**: [https://pan.baidu.com/s/1S1b4tV5VkzS0kroMGrcrHw?pwd=e4et](https://pan.baidu.com/s/1S1b4tV5VkzS0kroMGrcrHw?pwd=e4et)
- **Extraction Code**: `e4et`

*Note: Please place the downloaded `images` and `labels` folders into the corresponding subdirectories within the `dataset/` directory before training or evaluation.*
### Detection Categories
The model is trained to detect 4 typical tobacco leaf pathologies:
1. **Frogeye**
2. **Mosaic**
3. **Budworm damage**
4. **Wildfire**

## Implementation Details

- **Framework**: Built on Ultralytics RT-DETR.
- **Input Size**: $640 \times 640$.
- **Training Epochs**: 600.
- **Optimizer & Loss Function**:
    This project utilizes a multi-dimensional joint loss function for robust optimization. The core implementations can be found in `ultralytics/utils/loss.py`:
  - **SlideVariFocalLoss (SVFL)**: Custom implementation provided in this repository to address class imbalance.
  - **Focaler_MPDIoU**: Custom implementation provided in this repository for high-precision bounding box regression.
  - **L1 Loss**: Directly utilized from the **official PyTorch implementation** (`torch.nn.L1Loss`) for coordinate supervision.
- **Hardware**: Conducted on Ubuntu 22.04 with NVIDIA RTX A4000 GPUs.

## Citation

If you find this work helpful for your research, please cite our paper:

```bibtex
@article{zhao2026mcaddetr,
  title={MCAD-DETR: a real-time lightweight tobacco leaf disease and damage detection model},
  author={Zhao, Wenjun and Wu, Hao and Gao, Hui and Wang, Haofeng},
  journal={Journal of Real-Time Image Processing},
  year={2026},
  publisher={Springer}
}
```

## Contact

For questions and discussions, please contact the corresponding author:
**Hao Wu**: [wuhao@haue.edu.cn](mailto:wuhao@haue.edu.cn)

## License

This project is licensed under the **Apache License 2.0**. This allows for free use, modification, and distribution of the code for both academic and commercial purposes, provided that appropriate credit is given to the original authors.

---

**Note**: This repository contains the core implementation of the MCAD-DETR architecture. The complete training logs and pre-trained weights will be made public upon the official acceptance of the manuscript. Although we strive for accuracy, please be aware that minor differences in performance may occur due to environment hallucinations or hardware variations.