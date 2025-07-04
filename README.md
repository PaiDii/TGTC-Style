# TGTC-Style
This repository contains a pytorch implementation for the paper: Texture-Consistent 3D Scene Style Transfer via Transformer-Guided Neural Radiance Fields.
![consec](https://github.com/user-attachments/assets/aef659dc-3beb-49f0-90a9-2a5d6bfde95b)

---
## Environment

```bash
conda create -n tgtcstyle python=3.9  # (Python >= 3.8)
conda activate tgtcstyle
pip install -r requirements.txt
```

---
## Dataset and Pre-trained Models

Please download 3D scene datasets ([Baidu Disk](https://pan.baidu.com/s/1vhdMzwLt4QHdWycIWFbmIw?pwd=rkwp), code: `rkwp`) and Style image dataset ([Baidu Disk](https://pan.baidu.com/s/1ULNmkeURmCylJeSJDtP0ZA?pwd=ku49), code: `ku49`), then place the 3D scenes in `./data` and a style image in `./style`.

Please download pre-trained VGG, VAE, and CNN decoder ([Baidu Disk](https://pan.baidu.com/s/1BpWZYDauJwsse8QLSzTmfw?pwd=cvdg), code: `cvdg`), then place the VGG and VAE in `./pretrained` and the decoder in `./models`.

---
## Train

```bash
python run_stylenerf.py --config ./configs/fern.txt
```

---
## Evaluate

```bash
python run_stylenerf.py --config ./configs/fern.txt --render_train_style --chunk 1024
python run_stylenerf.py --config ./configs/fern.txt --render_valid_style --chunk 1024
```
