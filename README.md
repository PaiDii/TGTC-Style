# TGTC-Style
This repository contains a pytorch implementation for the paper: Texture-Consistent 3D Scene Style Transfer via Transformer-Guided Neural Radiance Fields.
![consec](https://github.com/user-attachments/assets/aef659dc-3beb-49f0-90a9-2a5d6bfde95b)

---
### Environment

```bash
conda create -n tgtcstyle python=3.9  # (Python >= 3.8)
conda activate tgtcstyle
pip install -r requirements.txt
```
---
### Dataset and Pre-trained Models
Please download 3D scene datasets ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q`) and Style image dataset ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q`), then put the 3D scenes in `./data` and put a style image in `./style`.

Please download pre-trained VGG ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q`), VAE ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q`), CNN decoder ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q`), then put the VGG and VAE in `./pretrained` and put the decoder in `./models`.

### Train
