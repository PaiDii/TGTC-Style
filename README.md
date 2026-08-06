<div align="center">

### Texture-Consistent 3D Scene Style Transfer via Transformer-Guided Neural Radiance Fields


[![Paper](https://img.shields.io/badge/Paper-IEEE%20TIP-00629B?style=for-the-badge\&logo=ieee)](https://doi.org/10.1109/TIP.2025.3626892)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.8-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Research%20Use-lightgrey?style=for-the-badge)](#license)

Official PyTorch implementation of the IEEE TIP 2025 paper:

If this project is helpful to your research, please consider giving the repository a ⭐.

</div>

---

<div align="center">
  <img src="https://github.com/user-attachments/assets/aef659dc-3beb-49f0-90a9-2a5d6bfde95b" width="100%" alt="TGTC-Style qualitative results">
</div>

---

## ✨ Overview

**TGTC-Style** is a texture-consistent 3D scene style-transfer framework built upon Neural Radiance Fields.

The method introduces transformer-guided feature modeling to transfer the visual appearance of a reference style image to a 3D scene while preserving scene structure and improving texture consistency across different viewpoints.

### Highlights

* **Texture-consistent stylization** for 3D scenes.
* **Transformer-guided feature modeling** for effective style representation.
* **Multi-view consistent rendering** under novel viewpoints.
* Support for both **training-view rendering** and **novel-view rendering**.
* End-to-end implementation based on **PyTorch** and **Neural Radiance Fields**.

---

## 📰 News

* **2025** — TGTC-Style was published in *IEEE Transactions on Image Processing*, Volume 34, pages 7193–7208.
* The training and rendering code is now available in this repository.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/PaiDii/TGTC-Style.git
cd TGTC-Style
```

### 2. Create the Conda environment

```bash
conda create -n tgtcstyle python=3.9
conda activate tgtcstyle
```

Python 3.9 is recommended, while Python 3.8 or later is supported.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Datasets and Pre-trained Models

### Datasets

Download the required datasets from the following links:

| Resource            | Download                                                               | Extraction Code | Target Directory |
| ------------------- | ---------------------------------------------------------------------- | --------------: | ---------------- |
| 3D scene datasets   | [Baidu Disk](https://pan.baidu.com/s/1vhdMzwLt4QHdWycIWFbmIw?pwd=rkwp) |          `rkwp` | `./data`         |
| Style image dataset | [Baidu Disk](https://pan.baidu.com/s/1ULNmkeURmCylJeSJDtP0ZA?pwd=ku49) |          `ku49` | `./style`        |

After downloading, organize the files as follows:

```text
TGTC-Style/
├── data/
│   └── fern/
│       ├── images/
│       ├── poses_bounds.npy
│       └── ...
│
└── style/
    └── style_image.jpg
```

Place each 3D scene dataset under `./data` and place the selected style image under `./style`.

### Pre-trained models

Download the pre-trained VGG, VAE, and CNN decoder models:

| Resource                  | Download                                                               | Extraction Code |
| ------------------------- | ---------------------------------------------------------------------- | --------------: |
| VGG, VAE, and CNN decoder | [Baidu Disk](https://pan.baidu.com/s/1BpWZYDauJwsse8QLSzTmfw?pwd=cvdg) |          `cvdg` |

Place the downloaded models in the following directories:

```text
TGTC-Style/
├── pretrained/
│   ├── VGG/
│   └── VAE/
│
└── models/
    └── decoder/
```

Specifically:

* Place the pre-trained **VGG** and **VAE** files in `./pretrained`.
* Place the pre-trained **CNN decoder** in `./models`.

Create these directories manually if they do not already exist:

```bash
mkdir -p data style pretrained models
```

---

## 📁 Repository Structure

A simplified directory layout is shown below:

```text
TGTC-Style/
├── configs/                 # Configuration files
├── data/                    # 3D scene datasets
├── style/                   # Reference style images
├── pretrained/              # Pre-trained VGG and VAE models
├── models/                  # Network modules and CNN decoder
├── requirements.txt         # Python dependencies
├── train_tgtcs.py           # Training and rendering entry point
└── README.md
```

---

## 🚀 Training

To train TGTC-Style on the `fern` scene, run:

```bash
python train_tgtcs.py \
    --config ./configs/fern.txt
```

The scene path, style image path, training parameters, and output settings can be modified in the corresponding configuration file.

For a new scene, create a configuration file under `./configs` and adjust the dataset and output paths accordingly.

---

## 🎨 Rendering

### Render stylized training views

```bash
python train_tgtcs.py \
    --config ./configs/fern.txt \
    --render_train_style \
    --chunk 1024
```

### Render stylized validation or novel views

```bash
python train_tgtcs.py \
    --config ./configs/fern.txt \
    --render_valid_style \
    --chunk 1024
```

The `--chunk` argument controls the number of rays processed simultaneously. Reduce this value when GPU memory is limited.

For example:

```bash
python train_tgtcs.py \
    --config ./configs/fern.txt \
    --render_valid_style \
    --chunk 512
```

---

## ⚙️ Configuration

The experiment settings are defined in configuration files under:

```text
./configs
```

For example:

```text
./configs/fern.txt
```

Before training or rendering, check the following settings in the configuration file:

* Scene dataset path.
* Reference style image path.
* Pre-trained model paths.
* Training output directory.
* Number of training iterations.
* Rendering resolution.
* Sampling and batch parameters.

---

## 💡 Usage Notes

* Make sure all dataset and model paths are consistent with the selected configuration file.
* Use a smaller `--chunk` value if CUDA out-of-memory errors occur during rendering.
* Each scene can use an independent configuration file.
* The final stylization results may depend on the selected reference style image and training settings.
* For reproducible experiments, keep the same dataset preprocessing and configuration parameters.

---

## 📖 Citation

Please cite our paper if you find this repository useful for your research:

```bibtex
@ARTICLE{TGTC-Style,
  author  = {Chen, Wudi and Zha, Zhiyuan and Wang, Shigang and
             Ali, Liaqat and Wen, Bihan and Yuan, Xin and
             Zhou, Jiantao and Zhu, Ce},
  journal = {IEEE Transactions on Image Processing},
  title   = {Texture-Consistent 3D Scene Style Transfer via
             Transformer-Guided Neural Radiance Fields},
  year    = {2025},
  volume  = {34},
  pages   = {7193--7208},
  doi     = {10.1109/TIP.2025.3626892}
}
```

---

## 🙏 Acknowledgements

We sincerely thank the authors and contributors of the open-source projects and datasets that supported this work.

This repository is intended for academic research and non-commercial use.

---

## 📬 Contact

For questions regarding the paper or code, please open an issue in this repository:

https://github.com/PaiDii/TGTC-Style/issues

---

## License

This code is provided for academic research purposes. Please refer to the repository license file for detailed terms of use.

<div align="center">

</div>
