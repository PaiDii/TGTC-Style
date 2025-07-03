import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import transformer as transformer
import StyTR as StyTR
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time
def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    print(type(size))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_img_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def transformer_render(
        content_dir,
        style_dir,
        output, # stylized_gen_4.0
        save_ext: str = '.png',
        content = None,
        style = None,
        vgg: str = './pretrained/vgg_normalised.pth',
        save_dir: str = './pretrained',
        position_embedding: str = 'sine',
        hidden_dim: int = 512,
        decoder_path: str = None,
        trans_path: str = None,
        embedding_path: str = None,
        content_size=512,
        style_size=512,
        crop=True,
        preserve_color=True
):
    output_path = output

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Either content or content_dir should be given.
    if content:
        content_paths = [Path(content)]
    else:
        content_dir_path = Path(content_dir)
        content_paths = [f for f in content_dir_path.glob('*')]

    # Either style or style_dir should be given.
    if style:
        style_paths = [Path(style)]
    else:
        style_dir_path = Path(style_dir)
        style_paths = [f for f in style_dir_path.glob('*')]

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    vgg_encoder = StyTR.vgg
    vgg_encoder.load_state_dict(torch.load(vgg))
    vgg_encoder = nn.Sequential(*list(vgg_encoder.children())[:31])

    decoder = StyTR.decoder
    # This line is preserved as requested.
    if decoder_path is not None:
        decoder.load_state_dict(torch.load(decoder_path))
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg_encoder.eval()
    embedding.eval()

    from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # ckpts = [os.path.join(save_dir, f) for f in sorted(os.listdir(save_dir)) if 'decoder' in f]
    # state_dict = torch.load(ckpts[-1])
    # for k, v in state_dict.items():
    #     namekey = k
    #     new_state_dict[namekey] = v
    # decoder.load_state_dict(new_state_dict['decoder'])

    new_state_dict = OrderedDict()
    ckpts = [os.path.join(save_dir, f) for f in sorted(os.listdir(save_dir)) if 'transformer' in f]
    state_dict = torch.load(ckpts[-1])
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    Trans.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    ckpts = [os.path.join(save_dir, f) for f in sorted(os.listdir(save_dir)) if 'embedding' in f]
    state_dict = torch.load(ckpts[-1])
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    network = StyTR.StyTrans(vgg_encoder, decoder, embedding, Trans)
    network.eval()
    network.to(device)

    style_name = {splitext(basename(style_paths[0]))[0]: 0}
    # Ensure style_dir is a valid path for this operation
    style_path_str = style_dir if style_dir else os.path.dirname(style)
    style_path_for_list = [os.path.join(style_path_str, f) for f in sorted(os.listdir(style_path_str))][0]
    style_img = np.moveaxis(style_img_transform()(Image.open(str(style_path_for_list))).unsqueeze(0).numpy(), 1, -1)
    style_feature = np.zeros([1, 1024], dtype=np.float32)

    for content_path in content_paths:
        for style_path in style_paths:
            print(content_path)

            content_tf1 = content_transform()
            content_tensor = content_tf1(Image.open(content_path).convert("RGB"))

            c, h, w = np.shape(content_tensor)
            style_tf1 = style_transform(h, w)
            style_tensor = style_tf1(Image.open(style_path).convert("RGB"))

            style_tensor = style_tensor.to(device).unsqueeze(0)
            content_tensor = content_tensor.to(device).unsqueeze(0)

            with torch.no_grad():
                output_tensor, sty_fea = network(content_tensor, style_tensor)
            output_tensor = output_tensor.cpu()

            output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                output_path, splitext(basename(content_path))[0],
                splitext(basename(style_path))[0], save_ext
            )
            resample_layer = nn.Upsample(size=(int(h), int(w)), mode='bilinear', align_corners=True)
            output_tensor = resample_layer(output_tensor)
            save_image(output_tensor, output_name)

            style_feature = np.append(style_feature, [np.concatenate([sty_fea.reshape(-1, 512).mean(dim=0).cpu().numpy(), sty_fea.reshape([-1, 512]).var(dim=0).cpu().numpy()])], axis=0)

    style_feature = np.sum(style_feature, axis=0, keepdims=True) / (style_feature.shape[0] - 1)
    np.savez(os.path.join(output_path, 'stylized_data'), style_names=style_name, style_paths=style_path_for_list, style_images=style_img, style_features=style_feature)


