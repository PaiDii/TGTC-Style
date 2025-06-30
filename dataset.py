import os
from builtins import print
from sys import exit

import cv2
import glob
import torch
import VGGNet
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from pathlib import Path
from torchvision import transforms, models
from torch.utils.data import Dataset
from Style_function import adaptive_instance_normalization
from ray_utils import batchified_get_rays
# import utils_jt
from load_mip360_v2 import load_nerf_360_v2_data
from load_tnt import load_tnt_data
from load_llff import load_llff_data
from load_blender import load_blender_data
from load_mip360 import load_mip_data
from nsvf_dataset import NSVFDataset
device = "cuda:1" if torch.cuda.is_available() else "cpu"
# 先把离散的视角合成，然后减少相机帧
def view_synthesis(cps, factor=10):
    frame_num = cps.shape[0]
    cps = np.array(cps)
    from scipy.spatial.transform import Slerp
    from scipy.spatial.transform import Rotation as R
    from scipy import interpolate as intp
    # rots：每张图的c2w的旋转矩阵
    rots = R.from_matrix(cps[:, :3, :3])
    # slerp：构建插值器对象，确定对应的图片帧和旋转矩阵的旋转插值
    slerp = Slerp(np.arange(frame_num), rots)
    # tran：每张图的平移向量
    tran = cps[:, :3, -1]
    # 构建插值器对象，确定对应的图片帧和平移向量一维线形插值
    f_tran = intp.interp1d(np.arange(frame_num), tran.T)

    new_num = int(frame_num * factor)
    # 将减少的帧数旋转插值得到新的少图像帧的旋转矩阵，平移向量一样
    new_rots = slerp(np.linspace(0, frame_num - 1, new_num)).as_matrix()
    new_trans = f_tran(np.linspace(0, frame_num - 1, new_num)).T

    new_cps = np.zeros([new_num, 4, 4], np.float)
    new_cps[:, :3, :3] = new_rots
    new_cps[:, :3, -1] = new_trans
    new_cps[:, 3, 3] = 1
    # 目的：获得减少图像帧所对应的合成相机位姿
    # 单纯的将相机位姿减少可能会导致视角不一致，先进行多视角的合成得到旋转插值函数再减少帧通过插值函数改变对应的旋转矩阵以及平移向量
    return new_cps

# 根据size参数设置对图像进行大小调整和裁剪，并将图像转换为张量。
def image_transform(size, crop=False):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    # crop是否进行中心裁剪
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    # 使用 transforms.Compose 将 transform_list 转换为操作序列 transform。
    transform = transforms.Compose(transform_list)
    # 函数的返回值为一个 transforms.Compose 对象，该对象是由一系列图像转换操作组成的操作序列。
    return transform

def get_features(tensor, layers=[22]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = models.vgg19(pretrained=True)
    # vgg.load_state_dict(torch.load('./pretrained/vgg19.pth'), strict=False)
    for param in vgg.parameters():
        param.requires_grad = False
    vgg = vgg.eval().to(device)
    outputs = []
    # final_ix = max(layers)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x = normalize(tensor)
    for name, layer in enumerate(vgg.features):
        x = layer(x)
        if name in layers:
            outputs.append(x)
            break
    return torch.tensor(outputs)

def random_crop_batch_tensor(batch_tensor, crop_size=256):
    B, C, H, W = batch_tensor.shape
    if H < crop_size or W < crop_size:
        raise ValueError(f"Input size ({H},{W}) is smaller than crop size {crop_size}")
    cropped = []
    for i in range(B):
        top = torch.randint(0, H - crop_size + 1, (1,)).item()
        left = torch.randint(0, W - crop_size + 1, (1,)).item()
        crop = batch_tensor[i, :, top:top + crop_size, left:left + crop_size]
        cropped.append(crop)

    return torch.stack(cropped)  # shape: [B, C, crop_size, crop_size]

from torch.utils.data import Dataset, DataLoader

class StyleTransferDataset(Dataset):
    def __init__(self, content_tensor, style_tensor, crop_size=256):
        self.content = content_tensor  # shape: (B, C, H, W)
        self.style = style_tensor      # shape: (B, C, H, W)
        self.crop_size = crop_size

        assert self.content.shape[0] == self.style.shape[0], "Content and style must have same batch size."

    def __len__(self):
        return self.content.shape[0]

    def __getitem__(self, idx):
        content_img = self.content[idx].unsqueeze(0)  # shape: (1, C, H, W)
        style_img = self.style[idx].unsqueeze(0)

        content_crop = random_crop_batch_tensor(content_img, crop_size=self.crop_size)[0]
        style_crop = random_crop_batch_tensor(style_img, crop_size=self.crop_size)[0]

        return content_crop, style_crop

def stylize_large_image(network, content_tensor, style_tensor, crop_size=256, stride=128):
    B, C, H, W = content_tensor.shape

    output_tensor = torch.zeros_like(content_tensor)
    weight_mask = torch.zeros_like(content_tensor)

    for i in range(0, H, stride):
        for j in range(0, W, stride):
            h_end = min(i + crop_size, H)
            w_end = min(j + crop_size, W)
            h_start = max(h_end - crop_size, 0)
            w_start = max(w_end - crop_size, 0)

            content_crop = content_tensor[:, :, h_start:h_end, w_start:w_end]

            if style_tensor.shape[2:] == content_tensor.shape[2:]:
                style_crop = style_tensor[:, :, h_start:h_end, w_start:w_end]
            else:
                style_crop = style_tensor  # style可能是global的，直接用

            with torch.no_grad():
                out_crop, _, _, _, _ = network(content_crop, style_crop)

            output_tensor[:, :, h_start:h_end, w_start:w_end] += out_crop
            weight_mask[:, :, h_start:h_end, w_start:w_end] += 1.0

    output_tensor /= weight_mask
    return output_tensor

def style_transfer_test(vgg, decoder, content, style, crop_size=256, stride=128):
    Trans = vit.Transformer()
    embedding = vit.PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg.eval()
    embedding.eval()

    network = vit.StyTrans(vgg, decoder, embedding, Trans)
    network = nn.DataParallel(network, device_ids=[0, 1])
    network.to(device)
    network.eval()

    ckpts = [os.path.join('./data/fern/', f) for f in sorted(os.listdir('./data/fern/')) if 'transformer' in f]
    state_dict = torch.load(ckpts[-1])
    network.module.transformer.load_state_dict(state_dict['transformer'])
    # network.decode.load_state_dict(state_dict['decoder'])
    network.module.embedding.load_state_dict(state_dict['embedding'])

    output = stylize_large_image(network, content, style, crop_size=crop_size, stride=stride)
    return output.cpu()

from torchvision.utils import save_image
import vit
def style_transfer(vgg, decoder, content, style, alpha=1.0, num_train_view=6, style_ori=None, interpolation_weights=None, return_feature=False):
    assert (0.0 <= alpha <= 1.0)
    # content_f = vgg(content)
    # style_f = vgg(style)
    embedding = vit.PatchEmbed()
    Trans = vit.Transformer()
    network = vit.StyTrans(vgg, decoder, embedding, Trans)
    network.train()
    network.to(device)
    network = nn.DataParallel(network, device_ids=[0, 1])

    step = 0
    ckpts = [os.path.join('./data/fern/', f) for f in sorted(os.listdir('./data/fern/')) if 'transformer' in f]
    if len(ckpts) > 0:
        ld_dict = torch.load(ckpts[-1])
        step = ld_dict['step']
        print('load ckpt, step:', step)

    optimizer = torch.optim.Adam([
        {'params': network.module.transformer.parameters()},
        # {'params': network.module.decode.parameters()},
        {'params': network.module.embedding.parameters()},
    ], lr=5e-4)
    content_pro = torch.movedim(torch.from_numpy(content).float(), -1, 1).to(device)
    print('content_shape, style_shape:', content_pro.shape, style.shape)
    # interval = int(content.shape[0] / num_train_view) + 1
    # print('intervl:', interval)
    # num_large_patch = 0
    tmp_content = torch.zeros((num_train_view, content_pro.shape[1], content_pro.shape[2], content_pro.shape[3])).to(device)
    # tmp_content = content_pro[:num_train_view]

    # dataset = StyleTransferDataset(content_tensor=content_pro, style_tensor=style, crop_size=256)
    # data_iter = iter(DataLoader(dataset, batch_size=num_train_view, shuffle=False, sampler=, num_workers=4))

    for i in tqdm(range(step, 101)):
        # tmp = i % interval
        # if tmp == 0:
        #     num_large_patch += 1
        tmp = i % content_pro.shape[0]

        for j in range(num_train_view):
            tmp_content[j] = content_pro[(tmp+j) % content_pro.shape[0]]
            # print('view_num:', (tmp+j) % content_pro.shape[0])

        content_tr = random_crop_batch_tensor(tmp_content, crop_size=256)
        style_tr = random_crop_batch_tensor(style, crop_size=256)

        # batch_data = next(data_iter)
        # content_tr, style_tr = batch_data[0].to(device), batch_data[1].to(device)
        out, loss_c, loss_s, l_identity1, l_identity2 = network(content_tr, style_tr)

        if i % 100 == 0:
            output_name = '{:s}/fern/1/{:s}{:s}'.format(
                './data', str(i), ".jpg"
            )
            out = torch.cat((content_tr, out), 0)
            out = torch.cat((style_tr, out), 0)
            save_image(out, output_name)

        if i % 100 == 0:
            output_name = '{:s}/fern/1/{:s}{:s}'.format('./data', str(i), "_train.jpg")
            out_full = stylize_large_image(network, content_pro, style_ori, crop_size=256, stride=128)
            save_image(out_full, output_name)  # Saving the full image

        loss_c = 7.0 * loss_c
        loss_s = 10.0 * loss_s
        loss = loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1)

        if i % 200 == 0:
            print(loss.sum().cpu().detach().numpy(), "-content:", loss_c.sum().cpu().detach().numpy(), "-style:",
                  loss_s.sum().cpu().detach().numpy()
                  , "-l1:", l_identity1.sum().cpu().detach().numpy(), "-l2:", l_identity2.sum().cpu().detach().numpy()
                  )

        optimizer.zero_grad()
        loss.sum().backward()
        # torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        optimizer.step()

        if i % 100 == 0 or i == 60001:
            sv_dict = {'transformer': network.module.transformer.state_dict(), 'embedding': network.module.embedding.state_dict(), 'step': i}
            torch.save(sv_dict, '{:s}/transformer_iter_{:d}.pth'.format('./data/fern/', i))
            # torch.save(sv_dict, '{:s}/transformer_iter_{:d}.pth'.format('./data/fern/', i))
            # torch.save(network.module.decode.state_dict(), '{:s}/decoder_iter_{:d}.pth'.format('./data/fern/', i))
            # torch.save(network.module.embedding.state_dict(), '{:s}/embedding_iter_{:d}.pth'.format('./data/fern/', i))

    out = style_transfer_test(vgg, decoder, content_pro, style_ori)
    print('stylzied output:', out.shape)
    output_name = '{:s}/fern/1/test/{:s}{:s}'.format('./data', str(1), ".jpg")
    save_image(out, output_name)

    # network.eval()
    # for i in tqdm(range(content_pro.shape[0])):
    #     with torch.no_grad():
    #         out, _, _, _, _ = network(content_pro[0:4, :, :256, :256], style_ori[0:4, :, :256, :256])
    #     out = out.cpu()
    #     output_name = '{:s}/fern/1/test/{:s}{:s}'.format('./data', str(i), ".jpg")
    #     save_image(out, output_name)

    # interval = int(content.shape[0] / num_train_view) + 1
    # print('intervl:', interval)
    # num_large_patch = 0
    # tmp_content = torch.zeros((num_train_view, content_pro.shape[1], content_pro.shape[2], content_pro.shape[3])).to(device)
    #
    # for i in tqdm(range(step, 50000)):
    #     # tmp = i % interval
    #     # if tmp == 0:
    #     #     num_large_patch += 1
    #     tmp = i % content_pro.shape[0]
    #
    #     for j in range(num_train_view):
    #         tmp_content[j] = content_pro[(tmp+j) % content_pro.shape[0]]
    #     # print('view_num:', (tmp+j) % content_pro.shape[0])
    #
    #     content_tr = random_crop_batch_tensor(tmp_content, crop_size=256)
    #     style_tr = random_crop_batch_tensor(style, crop_size=256)
    #     out, loss_c, loss_s, l_identity1, l_identity2 = network(content_tr, style_tr)
    #
    #     if i % 100 == 0:
    #         output_name = '{:s}/fern/1/{:s}{:s}'.format(
    #             './data', str(i), ".jpg"
    #         )
    #         out = torch.cat((content_tr, out), 0)
    #         out = torch.cat((style_tr, out), 0)
    #         print('stylized shape', out.shape)
    #         save_image(out, output_name)
    #
    #     loss_c = 7.0 * loss_c
    #     loss_s = 10.0 * loss_s
    #     loss = loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1)
    #
    #     if (i + 1) % 200 == 0:
    #         print(loss.sum().cpu().detach().numpy(), "-content:", loss_c.sum().cpu().detach().numpy(), "-style:",
    #               loss_s.sum().cpu().detach().numpy()
    #               , "-l1:", l_identity1.sum().cpu().detach().numpy(), "-l2:",
    #               l_identity2.sum().cpu().detach().numpy()
    #               )
    #
    #     optimizer.zero_grad()
    #     loss.sum().backward()
    #     # torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
    #     optimizer.step()
    #
    #     if (i + 1) % 500 == 0 or (i + 1) == 50000:
    #         state_dict = network.module.state_dict()
    #         for key in state_dict.keys():
    #             state_dict[key] = state_dict[key].to(torch.device('cpu'))
    #         sv_dict = {'transformer': state_dict, 'step': (i + 1)}
    #         torch.save(sv_dict, '{:s}/transformer_iter_{:d}.pth'.format('./data/fern/', i + 1))


    exit(0)
    return #torch.clamp(decoder(feat), 0., 1.), feat, style_f

# 2D风格迁移，获得adain后的风格图
def style_transfer_adain(vgg, decoder, content, style, alpha=1.0, interpolation_weights=None, return_feature=False):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    # content_f = get_features(content)
    # style_f = get_features(style)
    # interpolation_weight：插值权重是一组权重值，用于对归一化后的内容特征和风格特征进行线性插值。
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(content.device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            # 实现对同一风格的不同特征进行融合
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    # 根据权重 alpha 将插值特征与内容特征进行加权融合，得到最终的特征表示。
    feat = feat * alpha + content_f * (1 - alpha)
    if not return_feature:
        return decoder(feat)
    else:
        # torch.clamp(decoder(feat), 0., 1.) 的作用是将解码器生成的图像张量 decoder(feat) 中的数值限制在 [0, 1] 的范围内。
        # 具体而言，decoder(feat) 生成的图像张量若小于 0 则变成0或若大于 1 则变成1的数值，而将其限制在 [0, 1] 的范围内可以确保生成的图像符合图像的亮度范围。
        # 这在风格迁移任务中是常见的操作，以避免图像出现过亮或过暗的情况。
        return torch.clamp(decoder(feat), 0., 1.), feat, style_f


def style_data_prepare(style_path, content_images, size=512, chunk=64, sv_path=None, decode_path='./pretrained/decoder.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """VGG and Decoder"""
    decoder = VGGNet.decoder
    vgg = VGGNet.vgg
    decoder.eval()
    vgg.eval()
    print('Load decoder from ', decode_path)
    # 加载Decoder模型的权重
    decoder_data = torch.load(decode_path)
    if 'decoder' in decoder_data.keys():
        # 获得参数
        decoder.load_state_dict(decoder_data['decoder'])
    else:
        decoder.load_state_dict(decoder_data)
    # 加载预训练的VGG模型的权重
    vgg.load_state_dict(torch.load('./pretrained/vgg_normalised.pth'))
    # 创建VGG模型的子模型，仅保留前31层，这样做是为了获取VGG模型的前31层输出，用于计算风格特征。
    # *的作用是将一个可迭代对象（如列表、元组）解包为函数的参数。在这个例子中，*list(vgg.children())[:31] 将前31个子模块作为参数传递给 nn.Sequential() 函数。
    vgg = nn.Sequential(*list(vgg.children())[:31])
    # for param in vgg.parameters():
        # param.requires_grad = False
    # style_net = VGGNet.Net(vgg, decoder)
    # style_net.eval()
    # style_net.to(device)

    # vgg.to(device)
    # decoder.to(device)

    # 使用glob模块匹配指定目录下的.png、.jpg、.jpeg、.JPG和.PNG文件
    images_path = glob.glob(style_path + '/*.png') + glob.glob(style_path + '/*.jpg') + glob.glob(style_path + '/*.jpeg') + glob.glob(style_path + '/*.JPG') + glob.glob(style_path + '/*.PNG')
    # style_path：风格文件夹路径，images_path：风格图片路径
    print(style_path, images_path)

    style_images, style_paths, style_names = [], [], {}
    style_features = np.zeros([len(images_path), 1024], dtype=np.float32)
    style_img_features = np.zeros([len(images_path), 1024], dtype=np.float32)
    # 定义一个图像操作序列
    img_trans = image_transform((content_images.shape[1], content_images.shape[2]))

    # nst_net = NST_Net(encoder_pretrained_path='./pretrained/vgg_normalised2.pth')
    # 第i张风格图
    for i in tqdm(range(len(images_path))):
        images_path[i] = images_path[i].replace('\\', '/')
        print("Style Image: " + images_path[i])

        """Read Style Images"""
        style = img_trans(Image.open(images_path[i]))
        # np.moveaxis 函数将其通道维度移动到最后一个维度（从 (C, H, W) 转换为 (H, W, C)）
        # style_images: n*(512, 512, 3)（h，w，c）
        style_images.append(cv2.resize(np.moveaxis(style.numpy(), 0, -1), (512, 512)))
        # style_images = cv2.resize(np.moveaxis(style.numpy(), 0, -1), (512, 512))

        """Stylization"""
        stylized_images = np.zeros_like(content_images)
        style_feature = np.zeros([1, 1024], dtype=np.float32)
        style_img_feature = np.zeros([1, 1024], dtype=np.float32)
        # 在第0维度（批次维度）上添加一个维度。这是为了将单个样式图像转换为形状为 (1, H, W, C) 的张量
        # 将张量在指定的维度上进行扩展。这里通过扩展第0维度（批次维度）来生成一个形状为 (chunk, H, W, C) 的张量
        # 相当于chunk张style图片
        # style: [8, 3, 512, 512]
        num_train_view = 8 # original，num=4，2k轮次出结果
        style_ori = style.float().to(device).unsqueeze(0).expand([content_images.shape[0], *style.shape])
        style = style.float().to(device).unsqueeze(0).expand([num_train_view, *style.shape])
        print('type', type(style), content_images.shape[0])

        tmp_stylized_imgs, tmp_style_features, tmp_style_img_features = style_transfer(vgg=vgg, decoder=decoder, content=content_images, style=style[:content_images.shape[0]], alpha=1., style_ori=style_ori, num_train_view=num_train_view, return_feature=True)
        tmp_stylized_imgs = np.moveaxis(tmp_stylized_imgs.cpu().numpy(), 1, -1)
        stylized_images = cv2.resize(tmp_stylized_imgs, (stylized_images.shape[2], stylized_images.shape[1]))
        style_feature = np.append(style_feature, [np.concatenate([tmp_style_features[0].reshape(-1, 512).mean(dim=0).cpu().numpy(), tmp_style_features[0].reshape([-1, 512]).var(dim=0).cpu().numpy()])], axis=0)
        style_img_feature = np.append(style_img_feature, [np.concatenate([tmp_style_img_features[0].reshape(-1, 512).mean(dim=0).cpu().numpy(), tmp_style_img_features[0].reshape([-1, 512]).var(dim=0).cpu().numpy()])], axis=0)

        style_feature = np.sum(style_feature, axis=0) / (style_feature.shape[0] - 1)
        style_img_feature = np.sum(style_img_feature, axis=0) / (style_img_feature.shape[0] - 1)

        """Stylized Images Saving"""
        style_name = images_path[i].split('/')[-1].split('.')[0]
        style_names[style_name] = i
        if sv_path is not None:
            if not os.path.exists(sv_path + '/' + style_name):
                os.makedirs(sv_path + '/' + style_name)
            for j in range(stylized_images.shape[0]):
                Image.fromarray(np.array(stylized_images[j] * 255, np.uint8)).save(sv_path + '/' + style_name + '/%03d.png' % j)
                # if save_geo:
                #     # 保存为npz文件
                #     np.savez(sv_path + '/' + style_name + '/%03d' % j, stylized_image=stylized_images[j])
        style_paths.append(sv_path + '/' + style_name)
        style_features[i] = style_feature
        style_img_features[i] = style_img_feature
    # np.stack(style_images) 得到的新数组的形状将为 (n, H, W, C)
    style_images = np.stack(style_images)
    # n为风格图片的个数
    # style_names：风格图像名称，style_paths：风格化后照片储存的路径，style_images：所有风格图像(n，512, 512, 3), style_features：每张风格图像对所有内容图像的风格化后特征[n，1024], style_img_features：每张风格图像的特征[n，1024]
    return style_names, style_paths, style_images, style_features, style_img_features


# 风格数据准备，包括读取风格图片、进行风格迁移生成风格化图像，保存风格化图像，并提取风格特征
def style_data_prepare_original(style_path, content_images, size=512, chunk=64, sv_path=None, decode_path='./pretrained/decoder.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """VGG and Decoder"""
    decoder = VGGNet.decoder
    vgg = VGGNet.vgg
    decoder.eval()
    vgg.eval()
    print('Load decoder from ', decode_path)
    # 加载Decoder模型的权重
    decoder_data = torch.load(decode_path)
    if 'decoder' in decoder_data.keys():
        # 获得参数
        decoder.load_state_dict(decoder_data['decoder'])
    else:
        decoder.load_state_dict(decoder_data)
    # 加载预训练的VGG模型的权重
    vgg.load_state_dict(torch.load('./pretrained/vgg_normalised.pth'))
    # 创建VGG模型的子模型，仅保留前31层，这样做是为了获取VGG模型的前31层输出，用于计算风格特征。
    # *的作用是将一个可迭代对象（如列表、元组）解包为函数的参数。在这个例子中，*list(vgg.children())[:31] 将前31个子模块作为参数传递给 nn.Sequential() 函数。
    vgg = nn.Sequential(*list(vgg.children())[:31])
    # for param in vgg.parameters():
        # param.requires_grad = False
    # style_net = VGGNet.Net(vgg, decoder)
    # style_net.eval()
    # style_net.to(device)
    vgg.to(device)
    decoder.to(device)

    # 使用glob模块匹配指定目录下的.png、.jpg、.jpeg、.JPG和.PNG文件
    images_path = glob.glob(style_path + '/*.png') + glob.glob(style_path + '/*.jpg') + glob.glob(style_path + '/*.jpeg') + glob.glob(style_path + '/*.JPG') + glob.glob(style_path + '/*.PNG')
    # style_path：风格文件夹路径，images_path：风格图片路径
    print(style_path, images_path)

    style_images, style_paths, style_names = [], [], {}
    style_features = np.zeros([len(images_path), 1024], dtype=np.float32)
    style_img_features = np.zeros([len(images_path), 1024], dtype=np.float32)
    # 定义一个图像操作序列
    img_trans = image_transform(size)

    # nst_net = NST_Net(encoder_pretrained_path='./pretrained/vgg_normalised2.pth')
    # 第i张风格图
    for i in tqdm(range(len(images_path))):
        images_path[i] = images_path[i].replace('\\', '/')
        print("Style Image: " + images_path[i])

        """Read Style Images"""
        style = img_trans(Image.open(images_path[i]))
        # np.moveaxis 函数将其通道维度移动到最后一个维度（从 (C, H, W) 转换为 (H, W, C)）
        # style_images: n*(512, 512, 3)（h，w，c）
        style_images.append(cv2.resize(np.moveaxis(style.numpy(), 0, -1), (512, 512)))
        # style_images = cv2.resize(np.moveaxis(style.numpy(), 0, -1), (512, 512))

        """Stylization"""
        stylized_images = np.zeros_like(content_images)
        style_feature = np.zeros([1, 1024], dtype=np.float32)
        style_img_feature = np.zeros([1, 1024], dtype=np.float32)
        # 在第0维度（批次维度）上添加一个维度。这是为了将单个样式图像转换为形状为 (1, H, W, C) 的张量
        # 将张量在指定的维度上进行扩展。这里通过扩展第0维度（批次维度）来生成一个形状为 (chunk, H, W, C) 的张量
        # 相当于chunk张style图片
        # style: [8, 3, 512, 512]
        style = style.float().to(device).unsqueeze(0).expand([chunk, *style.shape])
        start = 0
        print('type', type(style), content_images.shape[0])
        while start < content_images.shape[0]:
            end = min(start + chunk, content_images.shape[0])
            # tmp_imgs：tensor（chunk，c，h，w）
            tmp_imgs = torch.movedim(torch.from_numpy(content_images[start: end]).float().to(device), -1, 1)
            # chunk*c*w*h
            # tmp_imgs = torch.movedim(tmp_imgs, -1, 2)
            with torch.no_grad():
                # _, _, tmp_stylized_imgs, tmp_style_features = style_net(content=tmp_imgs, style=style[:tmp_imgs.shape[0]], return_stylized_content = True)
                tmp_stylized_imgs, tmp_style_features, tmp_style_img_features = style_transfer_adain(vgg=vgg, decoder=decoder, content=tmp_imgs, style=style[:tmp_imgs.shape[0]], alpha=1., return_feature=True)
                # tmp_stylized_imgs：numpy（chunk，h，w，c）
                tmp_stylized_imgs = np.moveaxis(tmp_stylized_imgs.cpu().numpy(), 1, -1)
            for j in range(end-start):
                stylized_images[start+j] = cv2.resize(tmp_stylized_imgs[j], (stylized_images.shape[2], stylized_images.shape[1]))
            # tmp_style_features[0].reshape(-1, 512) 将第一个样本的特征重新调整为大小为 [n, 512] 的二维数组，其中 n 是特征数量。
            # .mean(dim=0) 表示在第一个维度上求平均值，即对每个特征维度求平均值，得到一个长度为 512 的平均特征向量，（512）
            # style_feature：（1024）
            # style_feature = np.concatenate([tmp_style_features[0].reshape(-1, 512).mean(dim=0).cpu().numpy(), tmp_style_features[0].reshape([-1, 512]).var(dim=0).cpu().numpy()])
            style_feature = np.append(style_feature, [np.concatenate([tmp_style_features[0].reshape(-1, 512).mean(dim=0).cpu().numpy(), tmp_style_features[0].reshape([-1, 512]).var(dim=0).cpu().numpy()])], axis=0)
            style_img_feature = np.append(style_img_feature, [np.concatenate([tmp_style_img_features[0].reshape(-1, 512).mean(dim=0).cpu().numpy(), tmp_style_img_features[0].reshape([-1, 512]).var(dim=0).cpu().numpy()])], axis=0)
            start = end
        style_feature = np.sum(style_feature, axis=0) / (style_feature.shape[0] - 1)
        style_img_feature = np.sum(style_img_feature, axis=0) / (style_img_feature.shape[0] - 1)

        """Stylized Images Saving"""
        style_name = images_path[i].split('/')[-1].split('.')[0]
        style_names[style_name] = i
        if sv_path is not None:
            if not os.path.exists(sv_path + '/' + style_name):
                os.makedirs(sv_path + '/' + style_name)
            for j in range(stylized_images.shape[0]):
                Image.fromarray(np.array(stylized_images[j] * 255, np.uint8)).save(sv_path + '/' + style_name + '/%03d.png' % j)
                # if save_geo:
                #     # 保存为npz文件
                #     np.savez(sv_path + '/' + style_name + '/%03d' % j, stylized_image=stylized_images[j])
        style_paths.append(sv_path + '/' + style_name)
        style_features[i] = style_feature
        style_img_features[i] = style_img_feature
    # np.stack(style_images) 得到的新数组的形状将为 (n, H, W, C)
    style_images = np.stack(style_images)
    # n为风格图片的个数
    # style_names：风格图像名称，style_paths：风格化后照片储存的路径，style_images：所有风格图像(n，512, 512, 3), style_features：每张风格图像对所有内容图像的风格化后特征[n，1024], style_img_features：每张风格图像的特征[n，1024]
    return style_names, style_paths, style_images, style_features, style_img_features

def prepare_adain(style_path, content_images, size=512, chunk=64, sv_path=None, decode_path='./pretrained/decoder.pth', save_geo=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """VGG and Decoder"""
    decoder = VGGNet.decoder
    vgg = VGGNet.vgg
    decoder.eval()
    vgg.eval()
    # 加载Decoder模型的权重
    decoder_data = torch.load(decode_path)
    if 'decoder' in decoder_data.keys():
        # 获得参数
        decoder.load_state_dict(decoder_data['decoder'])
    else:
        decoder.load_state_dict(decoder_data)
    # 加载预训练的VGG模型的权重
    vgg.load_state_dict(torch.load('./pretrained/vgg_normalised.pth'))
    # 创建VGG模型的子模型，仅保留前31层，这样做是为了获取VGG模型的前31层输出，用于计算风格特征。
    # *的作用是将一个可迭代对象（如列表、元组）解包为函数的参数。在这个例子中，*list(vgg.children())[:31] 将前31个子模块作为参数传递给 nn.Sequential() 函数。
    vgg = nn.Sequential(*list(vgg.children())[:31])
    # for param in vgg.parameters():
        # param.requires_grad = False
    # style_net = VGGNet.Net(vgg, decoder)
    # style_net.eval()
    # style_net.to(device)
    vgg.to(device)
    decoder.to(device)

    # 使用glob模块匹配指定目录下的.png、.jpg、.jpeg、.JPG和.PNG文件
    images_path = glob.glob(style_path + '/*.png') + glob.glob(style_path + '/*.jpg') + glob.glob(style_path + '/*.jpeg') + glob.glob(style_path + '/*.JPG') + glob.glob(style_path + '/*.PNG')
    # style_path：风格文件夹路径，images_path：风格图片路径
    print(style_path, images_path)

    style_images, style_paths, style_names = [], [], {}
    style_features = np.zeros([len(images_path), 1024], dtype=np.float32)
    style_img_features = np.zeros([len(images_path), 1024], dtype=np.float32)
    # 定义一个图像操作序列
    img_trans = image_transform(size)

    # nst_net = NST_Net(encoder_pretrained_path='./pretrained/vgg_normalised2.pth')
    # 第i张风格图
    for i in tqdm(range(len(images_path))):
        images_path[i] = images_path[i].replace('\\', '/')
        print("Style Image: " + images_path[i])

        """Read Style Images"""
        style = img_trans(Image.open(images_path[i]))
        # np.moveaxis 函数将其通道维度移动到最后一个维度（从 (C, H, W) 转换为 (H, W, C)）
        # style_images: n*(512, 512, 3)（h，w，c）
        style_images.append(cv2.resize(np.moveaxis(style.numpy(), 0, -1), (512, 512)))
        # style_images = cv2.resize(np.moveaxis(style.numpy(), 0, -1), (512, 512))

        """Stylization"""
        stylized_images = np.zeros_like(content_images)
        style_feature = np.zeros([1, 1024], dtype=np.float32)
        style_img_feature = np.zeros([1, 1024], dtype=np.float32)
        # 在第0维度（批次维度）上添加一个维度。这是为了将单个样式图像转换为形状为 (1, H, W, C) 的张量
        # 将张量在指定的维度上进行扩展。这里通过扩展第0维度（批次维度）来生成一个形状为 (chunk, H, W, C) 的张量
        # 相当于chunk张style图片
        # style: [8, 3, 512, 512]
        style = style.float().to(device).unsqueeze(0).expand([chunk, *style.shape])
        start = 0
        print('type', type(style))
        while start < content_images.shape[0]:
            end = min(start + chunk, content_images.shape[0])
            # tmp_imgs：tensor（chunk，c，h，w）
            tmp_imgs = torch.movedim(torch.from_numpy(content_images[start: end]).float().to(device), -1, 1)
            # chunk*c*w*h
            # tmp_imgs = torch.movedim(tmp_imgs, -1, 2)
            with torch.no_grad():
                # _, _, tmp_stylized_imgs, tmp_style_features = style_net(content=tmp_imgs, style=style[:tmp_imgs.shape[0]], return_stylized_content = True)
                tmp_stylized_imgs, tmp_style_features, tmp_style_img_features = style_transfer(vgg=vgg, decoder=decoder, content=tmp_imgs, style=style[:tmp_imgs.shape[0]], alpha=1., return_feature=True)
                # tmp_stylized_imgs：numpy（chunk，h，w，c）
                tmp_stylized_imgs = np.moveaxis(tmp_stylized_imgs.cpu().numpy(), 1, -1)
            for j in range(end-start):
                stylized_images[start+j] = cv2.resize(tmp_stylized_imgs[j], (stylized_images.shape[2], stylized_images.shape[1]))
            # tmp_style_features[0].reshape(-1, 512) 将第一个样本的特征重新调整为大小为 [n, 512] 的二维数组，其中 n 是特征数量。
            # .mean(dim=0) 表示在第一个维度上求平均值，即对每个特征维度求平均值，得到一个长度为 512 的平均特征向量，（512）
            # style_feature：（1024）
            # style_feature = np.concatenate([tmp_style_features[0].reshape(-1, 512).mean(dim=0).cpu().numpy(), tmp_style_features[0].reshape([-1, 512]).var(dim=0).cpu().numpy()])
            style_feature = np.append(style_feature, [np.concatenate([tmp_style_features[0].reshape(-1, 512).mean(dim=0).cpu().numpy(), tmp_style_features[0].reshape([-1, 512]).var(dim=0).cpu().numpy()])], axis=0)
            style_img_feature = np.append(style_img_feature, [np.concatenate([tmp_style_img_features[0].reshape(-1, 512).mean(dim=0).cpu().numpy(), tmp_style_img_features[0].reshape([-1, 512]).var(dim=0).cpu().numpy()])], axis=0)
            start = end
        style_feature = np.sum(style_feature, axis=0) / (style_feature.shape[0] - 1)
        style_img_feature = np.sum(style_img_feature, axis=0) / (style_img_feature.shape[0] - 1)

        """Stylized Images Saving"""
        style_name = images_path[i].split('/')[-1].split('.')[0]
        style_names[style_name] = i
        if sv_path is not None:
            if not os.path.exists(sv_path + '/' + style_name):
                os.makedirs(sv_path + '/' + style_name)
            for j in range(stylized_images.shape[0]):
                Image.fromarray(np.array(stylized_images[j] * 255, np.uint8)).save(sv_path + '/' + style_name + '/%03d.png' % j)
                if save_geo:
                    # 保存为npz文件
                    np.savez(sv_path + '/' + style_name + '/%03d' % j, stylized_image=stylized_images[j])
        style_paths.append(sv_path + '/' + style_name)
        style_features[i] = style_feature
        style_img_features[i] = style_img_feature
    # np.stack(style_images) 得到的新数组的形状将为 (n, H, W, C)
    style_images = np.stack(style_images)
    # n为风格图片的个数
    # style_names：风格图像名称，style_paths：风格化后照片储存的路径，style_images：所有风格图像(n，512, 512, 3), style_features：每张风格图像对所有内容图像的风格化后特征[n，1024], style_img_features：每张风格图像的特征[n，1024]
    return style_names, style_paths, style_images, style_features, style_img_features

# tensor版本
def get_rays(H, W, K, c2w, pixel_alignment=True):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    if pixel_alignment:
        i, j = i + .5, j + .5
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

# numpy版本
# 根据相机的内参、相机到世界的变换矩阵和图像的宽高，计算出每个像素对应的射线的起点和方向。
def get_rays_np(H, W, K, c2w, pixel_alignment=True):
    # 创建两个网格矩阵 i 和 j，表示图像平面上每个像素的横坐标和纵坐标。
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # 将网格矩阵 i 和 j 的值增加 0.5，以对齐像素的中心位置
    if pixel_alignment:
        i, j = i + .5, j + .5
    # dirs：射线的方向向量
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], axis=-1)
    # Rotate ray directions from camera frame to the world frame
    # 将射线方向向量从相机坐标系变换到世界坐标系。
    # 通过对射线方向向量与旋转矩阵的逐元素乘积，然后在最后一个维度求和，得到变换后的射线方向向量。
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 平移向量被用作射线的起点，平移向量（Translation Vector）是相机原点即光心起点到世界变换矩阵
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    # rays_d方向向量不用加平移向量
    # rays_o相机光心需要乘R加t但此时，相机坐标系下的光心坐标相当于（0,0,0），因此0*R=0,即相机光心就是t
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d

# "no_ndc" 在这个上下文中是指"no normalized device coordinates"（非标准化设备坐标）。
# 在计算机图形中，标准化设备坐标是指将物体的坐标范围映射到一个规范化的范围，通常是 [-1, 1] 或 [0, 1]。这个映射通常在投影步骤中进行，将物体坐标转换为屏幕空间坐标。
# 将射线从非标准化设备坐标（non-normalized device coordinates）转换为标准化设备坐标（normalized device coordinates）
def ndc_rays_np(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    # 将相机光心映射到场景近平面
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    # 通过对射线的起点和方向向量进行透视投影变换，可以将相机视野中的场景映射到标准化设备坐标空间
    # 在透视投影中，离观察者较远的对象会出现近似于平行线的效果，即在远处的物体相对于近处的物体显得较小。这种透视效果可以通过将射线的 x 分量除以 z 分量来实现。
    # 具体来说，对于每个射线，将 x 分量除以 z 分量可以确保远处的点（z 值较大）在投影后的坐标系中具有较小的 x 坐标值，而近处的点（z 值较小）在投影后的坐标系中具有较大的 x 坐标值。
    # 这样可以模拟出远处的物体相对于近处的物体较小的透视效果。
    # -1./(W/(2.*focal)) 是用于计算屏幕宽度与焦距之间比例的值，用于控制相机投影的透视效果和缩放因子。
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    # 正常rays_o是相机原点，现在rays_o变了，故射线方向要减去它
    d0 = -1./(W/(2.*focal)) * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = np.stack([o0, o1, o2], axis=-1)
    rays_d = np.stack([d0, d1, d2], axis=-1)

    return rays_o, rays_d

def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far

# 采样射线的类，获得相机光心到每个像素点的射线
class RaySampler(Dataset):
    def __init__(self, data_path, factor=2., mode='train', valid_factor=3, dataset_type='llff', white_bkgd=False, half_res=True, no_ndc=False, pixel_alignment=False, spherify=False, TT_far=4.):
        super().__init__()

        # K是相机的内参
        K = None
        if dataset_type == 'llff':
            images, poses, bds, render_poses, i_test = load_llff_data(data_path, factor, recenter=True, bd_factor=.75, spherify=spherify)
            print('images, poses, bds, render_poses, i_test:', images.shape, poses.shape, bds.shape, render_poses.shape)
            # print(poses[0])
            hwf = poses[0, :3, -1]
            # pose.shape=(图像数量，3, 4)
            poses = poses[:, :3, :4]
            # print(111111111, poses[0], '\n',render_poses[0])
            print('Loaded llff', images.shape, render_poses.shape, hwf, data_path)
            print('DEFINING BOUNDS')
            # "no_ndc" 在这个上下文中是指"no normalized device coordinates"（非标准化设备坐标）。
            # 在计算机图形中，标准化设备坐标是指将物体的坐标范围映射到一个规范化的范围，通常是 [-1, 1] 或 [0, 1]。这个映射通常在投影步骤中进行，将物体坐标转换为屏幕空间坐标。
            if no_ndc:
                # near：最近位置，far：最远位置
                near = np.ndarray.min(bds) * .9
                far = np.ndarray.max(bds) * 1.
            else:
                near = 0.
                far = 1.
            print('NEAR FAR', near, far)
        elif dataset_type == 'tnt':
            images, intrinsics, extrinsics, image_sizes, near, far, _, i_split, render_poses = load_tnt_data(data_path, "tnt", 1, 1, 1, 0.125, None, None)
            train_poses = render_poses[i_split[0]]
            K = intrinsics[i_split[0]][0]
            hwf = [image_sizes[0][0], image_sizes[0][1], K[0][0]]
            vaild_poses = render_poses[i_split[1]]
            images = images[i_split[0]]
            near, far = [0.01, 6]
            print('NEAR FAR', near, far)
        elif dataset_type == '1tnt':
            tntdataset = NSVFDataset(data_path, split="train", device=device, factor=1, n_images=None, dataset_type=dataset_type,
                                        epoch_size=12800, #  * args.__dict__.get('batch_size', 5000),
                                        scene_scale=None, scale=None, white_bkgd=True, normalize_by_bbox=False,
                                        data_bbox_scale=1.2, cam_scale_factor=0.95,# Truck:0.55,
                                        normalize_by_camera=True, permutation=False)
            train_poses = np.array(tntdataset.c2w)
            K = tntdataset.K
            hwf = [tntdataset.h_full, tntdataset.w_full, K[0][0]]
            near, far = [0.01, 6]

            valiad_tntdataset = NSVFDataset(data_path, split="test", device=device, factor=1, n_images=None,
                                     dataset_type=dataset_type,
                                     epoch_size=12800,  # * args.__dict__.get('batch_size', 5000),
                                     scene_scale=None, scale=None, white_bkgd=True, normalize_by_bbox=False,
                                     data_bbox_scale=1.2, cam_scale_factor=0.95,
                                     normalize_by_camera=True, permutation=False)
            vaild_poses = np.array(valiad_tntdataset.c2w)
            images = np.array(tntdataset.gt)
            images = np.concatenate((images, np.array(valiad_tntdataset.gt)), 0)

            # tntdataset = TanksTempleDataset(data_path, split='train', downsample=1.0, is_stack=True)
            # train_poses = tntdataset.poses.reshape(-1, 4, 4)
            # K = tntdataset.intrinsics
            # hwf = [tntdataset.img_wh[0], tntdataset.img_wh[1], K[0][0]]
            # near, far = tntdataset.near_far
            #
            # valiad_tntdataset = TanksTempleDataset(data_path, split='test', downsample=1.0, is_stack=True)
            # vaild_poses = valiad_tntdataset.poses.reshape(-1, 4, 4)
            # images = tntdataset.all_rgbs
            # images = np.concatenate((images, valiad_tntdataset.all_rgbs), 0)

            # images, poses, render_poses, hwf, K, i_split = load_and_rescale_tankstemple_data(data_path)
            # K = K.reshape(4, 4)
            # poses = poses.reshape(-1, 4, 4)
            # render_poses = render_poses.reshape(-1, 4, 4)
            # # images: [151, 546, 980, 3], poses: [151, 4, 4], render_poses: [18, 4, 4], hwf: [546, 980, 582.2704956224809], K: [4, 4]
            # print('Loaded tnt', images.shape, poses.shape, render_poses.shape, hwf)
            # i_train, i_test = i_split[0], i_split[1]
            # # near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)
            # near = 0.01
            # far = 6.
        elif dataset_type == 'blender':
            images, poses, render_poses, hwf, i_split = load_blender_data(data_path, testskip=4)
            # images: [150, 800, 800, 4], pose: [150, 4, 4], render_poses: [40, 4, 4]
            print('Loaded blender',type(images), images.shape, poses.shape, render_poses.shape, hwf)
            i_train, i_test = i_split
            near = 2.
            far = 6.
            if white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            else:
                images = images[..., :3]
        elif dataset_type == 'mip360':
            images, intrinsics, extrinsics, image_sizes, near, far, _, i_split, render_poses, render_poses_v, N_views1 = load_nerf_360_v2_data(data_path, "mip360", 4, 0.125, 1, 1, 1, None, None, False)
            train_poses = render_poses[i_split[0]]
            K = intrinsics[i_split[0]][0]
            hwf = [image_sizes[0][0], image_sizes[0][1], K[0][0]]
            vaild_poses = render_poses_v[:N_views1]
            images = images[i_split[0]]
            # near, far = [0.01, 6]
            print('NEAR FAR', near, far)
        elif dataset_type == '1mip360':
            images, poses, bds, render_poses, i_test = load_mip_data(data_path, factor, recenter=True, bd_factor=0.75, spherify=spherify)
            print('images, poses, bds, render_poses, i_test:', images.shape, poses.shape, bds.shape, render_poses.shape)
            # print(poses[0])
            hwf = poses[0, :3, -1]
            # pose.shape=(图像数量，3, 4)
            poses = poses[:, :3, :4]
            # print(111111111, poses[0], '\n', render_poses[0])
            print('Loaded mip360', images.shape, render_poses.shape, hwf, data_path)
            print('DEFINING BOUNDS')
            # "no_ndc" 在这个上下文中是指"no normalized device coordinates"（非标准化设备坐标）。
            # 在计算机图形中，标准化设备坐标是指将物体的坐标范围映射到一个规范化的范围，通常是 [-1, 1] 或 [0, 1]。这个映射通常在投影步骤中进行，将物体坐标转换为屏幕空间坐标。
            if no_ndc:
                # near：最近位置，far：最远位置
                near = np.ndarray.min(bds) * .9
                far = np.ndarray.max(bds) * 1.
            else:
                near = 0.
                far = 1.
            print('NEAR FAR', near, far)
        else:
            poses = hwf = K = near = far = None
            print('Unknown dataset type', dataset_type, 'exiting')
            exit(0)

        # 高，宽，焦距
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        # 内参矩阵
        # K 相机内参 focal 是焦距，0.5w 0.5h 是中心点坐标
        # 这个矩阵是相机坐标到图像坐标转换使用
        if K is None:
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

        """生成 Validation Rays"""
        # 获得验证用的rays
        if dataset_type == 'llff':
            # cps: 相机位姿，shape = (图像数量，4, 4)
            cps = np.concatenate([poses[:, :3, :4], np.zeros_like(poses[:, :1, :])], axis=1)
            cps[:, 3, 3] = 1.
            # cps_valid: [3*图像数量, 4, 4]
            cps_valid = np.concatenate([render_poses[:, :3, :4], np.zeros_like(render_poses[:, :1, :4])], axis=1)
            cps_valid[:, 3, 3] = 1.
            # cps_valid = view_synthesis(cps, valid_factor)
        elif dataset_type == 'tnt' or dataset_type == 'mip360':
            cps = train_poses
            cps_valid = vaild_poses
        else:
            cps = poses[i_train, :4, :4]
            cps_valid = poses[i_test, :4, :4]

        print('get rays of training and validation')
        # rays_o表示光线起点(相机中心在世界坐标系中的位置)，可以通过将相机坐标系的原点（通常是相机的光心）变换到世界坐标系来获得。
        # rays_d表示通过相机每个像素中心投射的每条光线的方向向量。在这种情况下，rays_o中的所有值都是相同的，因为该函数从单个摄影机获得光线。
        rays_o, rays_d = np.zeros([cps.shape[0], H, W, 3]), np.zeros([cps.shape[0], H, W, 3])
        for i in tqdm(range(cps.shape[0])):
            tmp_rays_o, tmp_rays_d = get_rays_np(H, W, K, cps[i, :3, :4], pixel_alignment)
            # 每张图的rays_d,rays_o
            rays_o[i] = tmp_rays_o
            rays_d[i] = tmp_rays_d
        rays_o_valid, rays_d_valid = np.zeros([cps_valid.shape[0], H, W, 3]), np.zeros([cps_valid.shape[0], H, W, 3])
        for i in tqdm(range(cps_valid.shape[0])):
            tmp_rays_o, tmp_rays_d = get_rays_np(H, W, K, cps_valid[i, :3, :4], pixel_alignment)
            rays_o_valid[i] = tmp_rays_o
            rays_d_valid[i] = tmp_rays_d

        if dataset_type == 'llff' and not no_ndc:
            rays_o, rays_d = ndc_rays_np(H, W, K[0][0], 1., rays_o, rays_d)
            rays_o_valid, rays_d_valid = ndc_rays_np(H, W, K[0][0], 1., rays_o_valid, rays_d_valid)

        print('K:', K)
        print('Camera Pose: ', cps.shape)

        if dataset_type == 'tnt' or dataset_type == 'mip360':
            rays_o, rays_d, _, _, _ = batchified_get_rays(
                intrinsics[i_split[0]],
                extrinsics[i_split[0]],
                image_sizes,
                True,
                False,
                False,
                False,
                None,
            )
            rays_o = torch.tensor(rays_o.reshape([cps.shape[0], H, W, 3]))
            rays_d = torch.tensor(rays_d.reshape([cps.shape[0], H, W, 3]))

            # rays_o = tntdataset.rays_o.reshape([cps.shape[0], H, W, 3])
            # rays_d = tntdataset.rays_d.reshape([cps.shape[0], H, W, 3])

        """Setting Attributes"""
        self.set_mode(mode)
        self.frame_num = cps.shape[0]
        self.h, self.w, self.f = H, W, focal
        self.hwf = hwf
        self.K = K
        self.cx, self.cy = W / 2., H / 2.
        self.cps, self.intr, self.images = cps, K, images
        self.cps_valid = cps_valid
        # 射线数量=所有图像像素
        self.rays_num = self.frame_num * self.h * self.w
        self.near, self.far = near, far
        self.rays_o, self.rays_d = rays_o, rays_d
        self.rays_o_valid, self.rays_d_valid = rays_o_valid, rays_d_valid

    def get_item_train(self, idx):
        frame_id = idx // (self.h * self.w)
        pixel_id = idx % (self.h * self.w)
        hid, wid = pixel_id // self.w, pixel_id % self.w
        rgb = self.images[frame_id, hid, wid]
        ray_o = self.rays_o[frame_id, hid, wid]
        ray_d = self.rays_d[frame_id, hid, wid]
        return {'rgb_gt': rgb, 'rays_o': ray_o, 'rays_d': ray_d}

    def get_item_valid(self, idx):
        frame_id = idx // (self.h * self.w)
        pixel_id = idx % (self.h * self.w)
        hid, wid = pixel_id // self.w, pixel_id % self.w
        ray_o = self.rays_o[frame_id, hid, wid]
        ray_d = self.rays_d[frame_id, hid, wid]
        return {'rays_o': ray_o, 'rays_d': ray_d}

    # 获取的样本是一个固定大小的图像块（patch），和get_item_train功能一样
    def get_patch_train(self, fid, hid, wid, patch_size=32):
        min_hid, min_wid = int(min(max(hid - patch_size / 2, 0), self.h - patch_size)), int(min(max(wid - patch_size / 2, 0), self.w - patch_size))
        max_hid, max_wid = min_hid + patch_size, min_wid + patch_size
        hids, wids = np.meshgrid(np.arange(min_hid, max_hid), np.arange(min_wid, max_wid))
        hids, wids = hids.reshape([-1]), wids.reshape([-1])
        rgbs = torch.from_numpy(np.stack([self.images[fid, hids[i], wids[i]] for i in range(hids.shape[0])], axis=0)).float()
        rays_o = torch.from_numpy(np.stack([self.rays_o[fid, hids[i], wids[i]] for i in range(hids.shape[0])], axis=0)).float()
        rays_d = torch.from_numpy(np.stack([self.rays_d[fid, hids[i], wids[i]] for i in range(hids.shape[0])], axis=0)).float()
        return {'rgb_gt': rgbs, 'ray_o': rays_o, 'rays_d': rays_d}

    def set_mode(self, mode='train'):
        modes = ['train', 'valid', 'train_style', 'valid_style']
        if mode not in modes:
            print('Unknown mode: ', mode, ' Only supports: ', modes)
            exit(-1)
        self.mode = mode

    def __getitem__(self, item):
        func_dict = {'train': self.get_item_train, 'valid': self.get_item_valid}
        return func_dict[self.mode](item)

    def __len__(self):
        if self.mode == 'train':
            return self.frame_num * self.w * self.h
        else:
            return self.cps_valid.shape[0] * self.w * self.h

# 采样射线+数据集内容图生成风格数据
class StyleRaySampler(Dataset):
    def __init__(self, data_path, style_path, factor=2., mode='train', valid_factor=3, dataset_type='llff', white_bkgd=False, half_res=True, no_ndc=False, pixel_alignment=False, spherify=False, TT_far=4.):
        super().__init__()
        K = None
        if dataset_type == 'llff':
            images, poses, bds, render_poses, i_test = load_llff_data(data_path, factor, recenter=True, bd_factor=.75, spherify=spherify)
            # print('images:'+images)
            hwf = poses[0, :3, -1]
            poses = poses[:, :3, :4]
            print('Loaded llff', images.shape, render_poses.shape, hwf, data_path)
            print('DEFINING BOUNDS')
            if no_ndc:
                near = np.ndarray.min(bds) * .9
                far = np.ndarray.max(bds) * 1.
            else:
                near = 0.
                far = 1.
            print('NEAR FAR', near, far)
        elif dataset_type == 'tnt':
            images, intrinsics, extrinsics, image_sizes, near, far, _, i_split, render_poses = load_tnt_data(data_path, "tnt", 1, 1, 1, 0.125, None, None)
            train_poses = render_poses[i_split[0]]
            K = intrinsics[i_split[0]][0]
            hwf = [image_sizes[0][0], image_sizes[0][1], K[0][0]]
            vaild_poses = render_poses[i_split[1]]
            images = images[i_split[0]]
            near, far = [0.01, 6]
            print('NEAR FAR', near, far)
        elif dataset_type == '1tnt':
            tntdataset = NSVFDataset(data_path, split="train", device=device, factor=1, n_images=None,
                                     dataset_type=dataset_type,
                                     epoch_size=12800,  # * args.__dict__.get('batch_size', 5000),
                                     scene_scale=None, scale=None, white_bkgd=True, normalize_by_bbox=False,
                                     data_bbox_scale=1.2, cam_scale_factor=0.95,
                                     normalize_by_camera=True, permutation=False)
            train_poses = np.array(tntdataset.c2w)
            K = tntdataset.K
            hwf = [tntdataset.h_full, tntdataset.w_full, K[0][0]]
            near, far = [0.01, 6]

            valiad_tntdataset = NSVFDataset(data_path, split="test", device=device, factor=1, n_images=None,
                                            dataset_type=dataset_type,
                                            epoch_size=12800,  # * args.__dict__.get('batch_size', 5000),
                                            scene_scale=None, scale=None, white_bkgd=True, normalize_by_bbox=False,
                                            data_bbox_scale=1.2, cam_scale_factor=0.95,
                                            normalize_by_camera=True, permutation=False)
            vaild_poses = np.array(valiad_tntdataset.c2w)
            images = np.array(tntdataset.gt)
            images = np.concatenate((images, np.array(valiad_tntdataset.gt)), 0)

            # tntdataset = TanksTempleDataset(data_path, split='train', downsample=1.0, is_stack=True)
            # train_poses = tntdataset.poses
            # K = tntdataset.intrinsics
            # hwf = [tntdataset.img_wh[0], tntdataset.img_wh[1], K[0][0]]
            # near, far = tntdataset.near_far
            #
            # valiad_tntdataset = TanksTempleDataset(data_path, split='test', downsample=1.0, is_stack=True)
            # vaild_poses = valiad_tntdataset.poses
            # images = tntdataset.all_rgbs
            # images = images.append(valiad_tntdataset.all_rgbs)

            # images, poses, render_poses, hwf, K, i_split = load_tankstemple_data(data_path)
            # K = K.reshape(4, 4)
            # poses = poses.reshape(-1, 4, 4)
            # render_poses = render_poses.reshape(-1, 4, 4)
            # # images: [151, 546, 980, 3], poses: [151, 4, 4], render_poses: [18, 4, 4], hwf: [546, 980, 582.2704956224809], K: [4, 4]
            # print('Loaded tnt', images.shape, render_poses.shape, hwf, i_split)
            # i_train, i_val, i_test = i_split
            # near = 0.01
            # far = 6.
        elif dataset_type == 'blender':
            images, poses, render_poses, hwf, i_split = load_blender_data(data_path, testskip=4)
            # images: [150, 800, 800, 4], pose: [150, 4, 4], render_poses: [40, 4, 4]
            print('Loaded blender', type(images), images.shape, poses.shape, render_poses.shape, hwf)
            i_train, i_test = i_split
            near = 2.
            far = 6.
            if white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            else:
                images = images[..., :3]
        elif dataset_type == 'mip360':
            images, intrinsics, extrinsics, image_sizes, near, far, _, i_split, render_poses, render_poses_v, N_views1 = load_nerf_360_v2_data(data_path, "mip360", 4, 0.125, 1, 1, 1, None, None, False)
            train_poses = render_poses[i_split[0]]
            K = intrinsics[i_split[0]][0]
            hwf = [image_sizes[0][0], image_sizes[0][1], K[0][0]]
            vaild_poses = render_poses_v[:N_views1]
            images = images[i_split[0]]
            # near, far = [0.01, 6]
            print('NEAR FAR', near, far)
        elif dataset_type == '1mip360':
            images, poses, bds, render_poses, i_test = load_mip_data(data_path, factor, recenter=True, bd_factor=0.75, spherify=spherify)
            print('images, poses, bds, render_poses, i_test:', images.shape, poses.shape, bds.shape, render_poses.shape)
            # print(poses[0])
            hwf = poses[0, :3, -1]
            # pose.shape=(图像数量，3, 4)
            poses = poses[:, :3, :4]
            # print(111111111, poses[0], '\n', render_poses[0])
            print('Loaded mip360', images.shape, render_poses.shape, hwf, data_path)
            print('DEFINING BOUNDS')
            # "no_ndc" 在这个上下文中是指"no normalized device coordinates"（非标准化设备坐标）。
            # 在计算机图形中，标准化设备坐标是指将物体的坐标范围映射到一个规范化的范围，通常是 [-1, 1] 或 [0, 1]。这个映射通常在投影步骤中进行，将物体坐标转换为屏幕空间坐标。
            if no_ndc:
                # near：最近位置，far：最远位置
                near = np.ndarray.min(bds) * .9
                far = np.ndarray.max(bds) * 1.
            else:
                near = 0.
                far = 1.
            print('NEAR FAR', near, far)
        else:
            poses = hwf = K = near = far = None
            print('Unknown dataset type', dataset_type, 'exiting')
            exit(0)

        # 高，宽，焦距
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        # 内参矩阵
        # K 相机内参 focal 是焦距，0.5w 0.5h 是中心点坐标
        # 这个矩阵是相机坐标到图像坐标转换使用
        if K is None:
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

        """生成 Validation Rays"""
        # 获得验证用的rays
        if dataset_type == 'llff':
            # cps: 相机位姿，shape = (图像数量，4, 4)
            cps = np.concatenate([poses[:, :3, :4], np.zeros_like(poses[:, :1, :])], axis=1)
            cps[:, 3, 3] = 1.
            # cps_valid: [3*图像数量, 4, 4]
            cps_valid = np.concatenate([render_poses[:, :3, :4], np.zeros_like(render_poses[:, :1, :4])], axis=1)
            cps_valid[:, 3, 3] = 1.
            # cps_valid = view_synthesis(cps, valid_factor)
        elif dataset_type == 'tnt' or dataset_type == 'mip360':
            cps = train_poses
            cps_valid = vaild_poses
        else:
            cps = poses[i_train, :4, :4]
            cps_valid = poses[i_test, :4, :4]

        print('get rays of training and validation')
        rays_o, rays_d = np.zeros([cps.shape[0], H, W, 3]), np.zeros([cps.shape[0], H, W, 3])
        for i in tqdm(range(cps.shape[0])):
            tmp_rays_o, tmp_rays_d = get_rays_np(H, W, K, cps[i, :3, :4], pixel_alignment)
            rays_o[i] = tmp_rays_o
            rays_d[i] = tmp_rays_d
        rays_o_valid, rays_d_valid = np.zeros([cps_valid.shape[0], H, W, 3]), np.zeros([cps_valid.shape[0], H, W, 3])
        for i in tqdm(range(cps_valid.shape[0])):
            tmp_rays_o, tmp_rays_d = get_rays_np(H, W, K, cps_valid[i, :3, :4], pixel_alignment)
            rays_o_valid[i] = tmp_rays_o
            rays_d_valid[i] = tmp_rays_d

        if (dataset_type == 'llff') and not no_ndc:
            rays_o, rays_d = ndc_rays_np(H, W, K[0][0], 1., rays_o, rays_d)
            rays_o_valid, rays_d_valid = ndc_rays_np(H, W, K[0][0], 1., rays_o_valid, rays_d_valid)
        # rays_o, rays_d: [20, 756, 1008, 3]
        # rays_o_valid, rays_d_valid: [20, 756, 1008, 3]

        if dataset_type == 'tnt' or dataset_type == 'mip360':
            rays_o, rays_d, _, _, _ = batchified_get_rays(
                intrinsics[i_split[0]],
                extrinsics[i_split[0]],
                image_sizes,
                True,
                False,
                False,
                False,
                None,
            )
            rays_o = torch.tensor(rays_o.reshape([cps.shape[0], H, W, 3]))
            rays_d = torch.tensor(rays_d.reshape([cps.shape[0], H, W, 3]))
            # rays_o = tntdataset.rays_o.reshape([cps.shape[0], H, W, 3])
            # rays_d = tntdataset.rays_d.reshape([cps.shape[0], H, W, 3])

        """Style Data"""
        # 生成data里的stylized_4.0
        if not os.path.exists(data_path + '/stylized_' + str(factor) + '/' + '/stylized_data.npz'):
            print("Stylizing training data ...")
            style_names, style_paths, style_images, style_features, _ = style_data_prepare(style_path, images, size=512, chunk=8, sv_path=data_path + '/stylized_' + str(factor) + '/', decode_path='./pretrained/decoder.pth')
            # 储存为npz，4个为keys
            np.savez(data_path + '/stylized_' + str(factor) + '/' + '/stylized_data', style_names=style_names, style_paths=style_paths, style_images=style_images, style_features=style_features)
        else:
            print("Stylized data from " + data_path + '/stylized_' + str(factor) + '/' + '/stylized_data.npz')
            stylized_data = np.load(data_path + '/stylized_' + str(factor) + '/' + '/stylized_data.npz', allow_pickle=True)
            style_names, style_paths, style_images, style_features = stylized_data['style_names'], stylized_data['style_paths'], stylized_data['style_images'], stylized_data['style_features']
            print("Dataset Creation Done!")

        """Setting Attributes"""
        self.set_mode(mode)
        self.frame_num = cps.shape[0]
        self.h, self.w, self.f = H, W, focal
        self.hwf = hwf
        self.K = K
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.cps, self.intr, self.images = cps, K, images
        self.cps_valid = cps_valid
        self.rays_num = self.frame_num * self.h * self.w
        self.near, self.far = near, far

        self.style_names = style_names
        self.style_images = style_images
        self.style_features = style_features
        self.style_paths = style_paths

        self.style_num = self.style_images.shape[0]
        self.rays_o, self.rays_d = rays_o, rays_d
        self.rays_o_valid, self.rays_d_valid = rays_o_valid, rays_d_valid

    def get_item_train(self, idx):
        frame_id = idx // (self.h * self.w)
        pixel_id = idx % (self.h * self.w)
        hid, wid = pixel_id // self.w, pixel_id % self.w
        rgb = self.images[frame_id, hid, wid]
        ray_o = self.rays_o[frame_id, hid, wid]
        ray_d = self.rays_d[frame_id, hid, wid]
        return {'rgb_gt': rgb, 'rays_o': ray_o, 'rays_d': ray_d}

    def get_item_train_style(self, idx):
        style_id = idx // (self.frame_num * self.h * self.w)
        frame_id = (idx % (self.frame_num * self.h * self.w)) // (self.h * self.w)
        hid = (idx % (self.h * self.w)) // self.w
        wid = idx % self.w
        stylized_contents = np.load(self.style_paths[style_id] + '/%03d.npz' % frame_id)['stylized_image']
        rgb = stylized_contents[hid, wid]
        rgb_origin = self.images[frame_id, hid, wid]
        style_feature = self.style_features[style_id]
        ray_o = self.rays_o[frame_id, hid, wid]
        ray_d = self.rays_d[frame_id, hid, wid]
        return {'rgb_gt': rgb, 'rays_o': ray_o, 'rays_d': ray_d, 'style_feature': style_feature, 'rgb_origin': rgb_origin, 'style_id': style_id, 'frame_id': frame_id}

    def get_item_valid(self, idx):
        frame_id = idx // (self.h * self.w)
        pixel_id = idx % (self.h * self.w)
        hid, wid = pixel_id // self.w, pixel_id % self.w
        ray_o = self.rays_o_valid[frame_id, hid, wid]
        ray_d = self.rays_d_valid[frame_id, hid, wid]
        return {'rays_o': ray_o, 'rays_d': ray_d}

    def get_item_valid_style(self, idx):
        style_id = idx // (self.cps_valid.shape[0] * self.h * self.w)
        frame_id = (idx % (self.cps_valid.shape[0] * self.h * self.w)) // (self.h * self.w)
        hid = (idx % (self.h * self.w)) // self.w
        wid = idx % self.w
        ray_o = self.rays_o_valid[frame_id, hid, wid]
        ray_d = self.rays_d_valid[frame_id, hid, wid]
        style_image = torch.from_numpy(self.style_images[style_id]).float()
        return {'rays_o': ray_o, 'rays_d': ray_d, 'style_image': style_image, 'style_id': style_id, 'frame_id': frame_id}

    def get_patch_train(self, fid, hid, wid, patch_size=32):
        min_hid, min_wid = int(min(max(hid - patch_size / 2, 0), self.h - patch_size)), int(min(max(wid - patch_size / 2, 0), self.w - patch_size))
        max_hid, max_wid = min_hid + patch_size, min_wid + patch_size
        hids, wids = np.meshgrid(np.arange(min_hid, max_hid), np.arange(min_wid, max_wid))
        hids, wids = hids.reshape([-1]), wids.reshape([-1])
        rgbs = torch.from_numpy(np.stack([self.images[fid, hids[i], wids[i]] for i in range(hids.shape[0])], axis=0)).float()
        rays_o = torch.from_numpy(np.stack([self.rays_o[fid, hids[i], wids[i]] for i in range(hids.shape[0])], axis=0)).float()
        rays_d = torch.from_numpy(np.stack([self.rays_d[fid, hids[i], wids[i]] for i in range(hids.shape[0])], axis=0)).float()
        return {'rgb_gt': rgbs, 'ray_o': rays_o, 'rays_d': rays_d}

    def get_patch_train_style(self, style_id, fid, min_hid, min_wid, patch_size=32):
        max_hid, max_wid = min_hid + patch_size, min_wid + patch_size
        rgbs_origin = torch.from_numpy(self.images[fid, min_hid: max_hid, min_wid: max_wid]).float()
        style_image = torch.from_numpy(self.style_images[style_id]).float()
        rays_o = torch.from_numpy(self.rays_o[fid, min_hid: max_hid, min_wid: max_wid]).float()
        rays_d = torch.from_numpy(self.rays_d[fid, min_hid: max_hid, min_wid: max_wid]).float()
        style_id = torch.tensor(style_id).expand([patch_size**2]).long()
        frame_id = torch.tensor(fid).expand([patch_size**2]).long()
        content_image = torch.from_numpy(self.images[frame_id]).float()
        return {'style_image': style_image, 'content_image': content_image, 'rays_o': rays_o, 'rays_d': rays_d, 'rgb_origin': rgbs_origin, 'style_id': style_id, 'frame_id': frame_id}

    def set_mode(self, mode='train'):
        modes = ['train', 'valid', 'train_style', 'valid_style']
        if mode not in modes:
            print('Unknown mode: ', mode, ' Only supports: ', modes)
            exit(-1)
        self.mode = mode

    def __getitem__(self, idx):
        func_dict = {'train': self.get_item_train, 'valid': self.get_item_valid, 'train_style': self.get_item_train_style, 'valid_style': self.get_item_valid_style}
        return func_dict[self.mode](idx)

    def __len__(self):
        if self.mode == 'train':
            return self.frame_num * self.w * self.h
        elif self.mode == 'valid':
            return self.cps_valid.shape[0] * self.w * self.h
        elif self.mode == 'train_style':
            return self.style_num * self.frame_num * self.w * self.h
        else:
            return self.style_num * self.cps_valid.shape[0] * self.w * self.h

# 通过像素索引获得射线起点和射线方向
# hid：垂直像素索引，wid: 水平像素索引，cx：图像中心点的水平坐标，cy: 图像中心点的垂直坐标
def get_rays_from_id(hid, wid, focal, c2w, cx=None, cy=None):
    # (wid - cx) 表示物体在图像平面上的水平位置相对于图像中心点的偏移量，而 (wid - cx) / focal 则表示了该偏移量相对于焦距的归一化大小。
    # 通过这样的归一化处理，可以消除焦距对物体在图像中位置的影响，使得不同焦距的相机或图像之间的位置坐标具有可比性。
    # 表示光线在相机坐标系中的深度方向为负一，这是因为在常规相机设置中，深度方向是指向相机内部的
    dir = np.stack([(wid - cx) / focal, - (hid - cy) / focal, -np.ones_like(wid)], axis=-1)
    # 表示将 c2w[:3, :3] 与 dir 进行乘积，并对结果按照指定的约定进行求和。
    ray_d = np.einsum('wc,c->w', c2w[:3, :3], dir)
    # 对光线的方向向量进行归一化处理，使其长度为1。它通过 np.linalg.norm 函数计算向量的长度，并将光线方向向量 ray_d 除以该长度，从而得到单位长度的光线方向向量。
    ray_d = ray_d / np.linalg.norm(ray_d)
    ray_o = c2w[:3, -1]
    ray_o, ray_d = np.array(ray_o, dtype=np.float32), np.array(ray_d, dtype=np.float32)
    return ray_o, ray_d

# 采样射线+nerf生成内容图生成风格数据
class StyleRaySampler_gen(Dataset):
    # gen_path：nerf生成的内容数据地址
    def __init__(self, data_path, style_path, gen_path, factor=2., mode='train', valid_factor=0.05, dataset_type='llff', white_bkgd=False, half_res=True,
                 no_ndc=False, pixel_alignment=False, spherify=False, decode_path='./pretrained/decoder.pth', store_rays=True, TT_far=4., collect_stylized_images=True):
        super().__init__()

        K = None
        if dataset_type == 'llff':
            images, poses, bds, render_poses, i_test = load_llff_data(data_path, factor, recenter=True, bd_factor=.75, spherify=spherify)
            # imagess = images
            hwf = poses[0, :3, -1]
            poses = poses[:, :3, :4]
            render_poses = render_poses[:, :3, :4]
            print('Loaded llff', images.shape, poses.shape, render_poses.shape, hwf, data_path)
            print('DEFINING BOUNDS')
            if no_ndc:
                near = np.ndarray.min(bds) * .9
                far = np.ndarray.max(bds) * 1.
            else:
                near = 0.
                far = 1.
            print('NEAR FAR', near, far)
        elif dataset_type == 'tnt':
            images, intrinsics, extrinsics, image_sizes, near, far, _, i_split, render_poses = load_tnt_data(data_path, "tnt", 1, 1, 1, 0.125, None, None)
            train_poses = render_poses[i_split[0]]
            K = intrinsics[i_split[0]][0]
            hwf = [image_sizes[0][0], image_sizes[0][1], K[0][0]]
            vaild_poses = render_poses[i_split[1]]
            images = images[i_split[0]]
            near, far = [0.01, 6]
            print('NEAR FAR', near, far)
        elif dataset_type == '1tnt':
            tntdataset = NSVFDataset(data_path, split="train", device=device, factor=1, n_images=None,
                                     dataset_type=dataset_type,
                                     epoch_size=12800,  # * args.__dict__.get('batch_size', 5000),
                                     scene_scale=None, scale=None, white_bkgd=True, normalize_by_bbox=False,
                                     data_bbox_scale=1.2, cam_scale_factor=0.95, # truck:0.55,
                                     normalize_by_camera=True, permutation=False)
            train_poses = np.array(tntdataset.c2w)
            K = tntdataset.K
            hwf = [tntdataset.h_full, tntdataset.w_full, K[0][0]]
            near, far = [0.01, 6]

            valiad_tntdataset = NSVFDataset(data_path, split="test", device=device, factor=1, n_images=None,
                                            dataset_type=dataset_type,
                                            epoch_size=12800,  # * args.__dict__.get('batch_size', 5000),
                                            scene_scale=None, scale=None, white_bkgd=True, normalize_by_bbox=False,
                                            data_bbox_scale=1.2, cam_scale_factor=0.95, # truck:0.55,
                                            normalize_by_camera=True, permutation=False)
            vaild_poses = np.array(valiad_tntdataset.c2w)
            images = np.array(tntdataset.gt)
            images = np.concatenate((images, np.array(tntdataset.gt)), 0)
            images_valid = np.array(valiad_tntdataset.gt)
            # images_valid = np.concatenate((images_valid, np.array(valiad_tntdataset.gt)), 0)

            # tntdataset = TanksTempleDataset(data_path, split='train', downsample=1.0, is_stack=True)
            # images = tntdataset.all_rgbs
            # imagess = images
            # train_poses = tntdataset.poses
            # K = tntdataset.intrinsics
            # hwf = [tntdataset.img_wh[0], tntdataset.img_wh[1], K[0][0]]
            # near, far = tntdataset.near_far
            #
            # valiad_tntdataset = TanksTempleDataset(data_path, split='test', downsample=1.0, is_stack=True)
            # vaild_poses = valiad_tntdataset.poses

            # images, poses, render_poses, hwf, K, i_split = load_tankstemple_data(data_path)
            # imagess = images
            # K = K.reshape(4, 4)
            # poses = poses.reshape(-1, 4, 4)
            # render_poses = render_poses.reshape(-1, 4, 4)
            # # images: [151, 546, 980, 3], poses: [151, 4, 4], render_poses: [18, 4, 4], hwf: [546, 980, 582.2704956224809], K: [4, 4]
            # print('Loaded tnt', images.shape, render_poses.shape, hwf, i_split)
            # i_train, i_val, i_test = i_split
            # near = 0.01
            # far = 6.
        elif dataset_type == 'blender':
            images, poses, render_poses, hwf, i_split = load_blender_data(data_path, testskip=4)
            # imagess = images
            # images: [150, 800, 800, 4], pose: [150, 4, 4], render_poses: [40, 4, 4]
            print('Loaded blender', type(images), images.shape, poses.shape, render_poses.shape, hwf)
            i_train, i_test = i_split
            near = 2.
            far = 6.
            if white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            else:
                images = images[..., :3]
        elif dataset_type == 'mip360':
            images, intrinsics, extrinsics, image_sizes, near, far, _, i_split, render_poses, render_poses_v, N_views1 = load_nerf_360_v2_data(data_path, "mip360", 4, 0.125, 1, 1, 1, None, None, False)
            # print(111, images.shape, render_poses.shape, i_split)
            train_poses = render_poses[i_split[0]]
            K = intrinsics[i_split[0]][0]
            hwf = [image_sizes[0][0], image_sizes[0][1], K[0][0]]
            vaild_poses = render_poses_v[:N_views1]
            images = images[i_split[0]]
            # near, far = [0.01, 6]
            print('NEAR FAR', near, far)
        elif dataset_type == '1mip360':
            images, poses, bds, render_poses, i_test = load_mip_data(data_path, factor, recenter=True, bd_factor=0.75, spherify=spherify)
            print('images, poses, bds, render_poses, i_test:', images.shape, poses.shape, bds.shape, render_poses.shape)
            # print(poses[0])
            hwf = poses[0, :3, -1]
            # pose.shape=(图像数量，3, 4)
            poses = poses[:, :3, :4]
            # print(111111111, poses[0], '\n', render_poses[0])
            print('Loaded mip360', images.shape, render_poses.shape, hwf, data_path)
            print('DEFINING BOUNDS')
            # "no_ndc" 在这个上下文中是指"no normalized device coordinates"（非标准化设备坐标）。
            # 在计算机图形中，标准化设备坐标是指将物体的坐标范围映射到一个规范化的范围，通常是 [-1, 1] 或 [0, 1]。这个映射通常在投影步骤中进行，将物体坐标转换为屏幕空间坐标。
            if no_ndc:
                # near：最近位置，far：最远位置
                near = np.ndarray.min(bds) * .9
                far = np.ndarray.max(bds) * 1.
            else:
                near = 0.
                far = 1.
            print('NEAR FAR', near, far)
        else:
            poses = hwf = K = near = far = None
            print('Unknown dataset type', dataset_type, 'exiting')
            exit(0)

        # calculate adain
        # _, _, _, _, _ = style_data_prepare(style_path, images_valid, size=512, chunk=8, sv_path=data_path + '/Adain_valid' + str(factor) + '/', decode_path=decode_path)
        # H, W, focal = hwf
        # images_valid = np.zeros([31, H, W, 3], np.float32)
        # for i in range(31):
        #     images_valid[i] = np.array(Image.open(str(sorted(list(Path('./logs/render_valid_120001/').glob('fine_*.png')))[i])).convert('RGB'), dtype=np.float32) / 255.
        # _, _, _, _, _ = style_data_prepare(style_path, images_valid, size=512, chunk=8, sv_path=data_path + '/Adain' + '/', decode_path=decode_path)

        # 高，宽，焦距
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        # 内参矩阵
        # K 相机内参 focal 是焦距，0.5w 0.5h 是中心点坐标
        # 这个矩阵是相机坐标到图像坐标转换使用
        if K is None:
            K = np.array([
                [focal, 0, 0.5 * W],
                [0, focal, 0.5 * H],
                [0, 0, 1]
            ])

        # 加载gen_path数据，读取照片、hwf、cps
        self.gen_path = gen_path
        self.image_paths = sorted(list(Path(self.gen_path).glob('rgb_*.png')))
        self.geo_paths = sorted(list(Path(self.gen_path).glob('geometry_*.npz')))
        data = np.load(str(self.geo_paths[0]))
        self.hwf = data['hwf']
        frame_num = len(self.image_paths)
        images = np.zeros([frame_num, H, W, 3], np.float32)
        cps = np.zeros([frame_num, 4, 4], np.float32)
        for i in range(frame_num):
            images[i] = np.array(Image.open(str(self.image_paths[i])).convert('RGB'), dtype=np.float32) / 255.
            cps[i] = np.load(str(self.geo_paths[i]))['cps']
        if white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])

        """生成 Validation Rays"""
        # 获得验证用的rays
        if dataset_type == 'llff':
            # cps: 相机位姿，shape = (图像数量，4, 4)
            cps = np.concatenate([poses[:, :3, :4], np.zeros_like(poses[:, :1, :])], axis=1)
            cps[:, 3, 3] = 1.
            # cps_valid: [3*图像数量, 4, 4]
            cps_valid = np.concatenate([render_poses[:, :3, :4], np.zeros_like(render_poses[:, :1, :4])], axis=1)
            cps_valid[:, 3, 3] = 1.
            # cps_valid = view_synthesis(cps, valid_factor)
        elif dataset_type == 'tnt' or dataset_type == 'mip360':
            cps = train_poses
            cps_valid = vaild_poses
        else:
            cps = poses[i_train, :4, :4]
            cps_valid = poses[i_test, :4, :4]

        if store_rays:
            print('get rays of training and validation')
            rays_o, rays_d = np.zeros([cps.shape[0], H, W, 3]), np.zeros([cps.shape[0], H, W, 3])
            for i in tqdm(range(cps.shape[0])):
                tmp_rays_o, tmp_rays_d = get_rays_np(H, W, K, cps[i, :3, :4], pixel_alignment)
                rays_o[i] = tmp_rays_o
                rays_d[i] = tmp_rays_d
            rays_o_valid, rays_d_valid = np.zeros([cps_valid.shape[0], H, W, 3]), np.zeros([cps_valid.shape[0], H, W, 3])
            for i in tqdm(range(cps_valid.shape[0])):
                tmp_rays_o, tmp_rays_d = get_rays_np(H, W, K, cps_valid[i, :3, :4], pixel_alignment)
                rays_o_valid[i] = tmp_rays_o
                rays_d_valid[i] = tmp_rays_d

            if (dataset_type == 'llff') and not no_ndc:
                rays_o, rays_d = ndc_rays_np(H, W, K[0][0], 1., rays_o, rays_d)
                rays_o_valid, rays_d_valid = ndc_rays_np(H, W, K[0][0], 1., rays_o_valid, rays_d_valid)
        else:
            rays_o, rays_d, rays_o_valid, rays_d_valid = None, None, None, None

        if dataset_type == 'tnt':
            rays_o, rays_d, _, _, _ = batchified_get_rays(
                intrinsics[i_split[0]],
                extrinsics[i_split[0]],
                image_sizes,
                True,
                False,
                False,
                False,
                None,
            )
            rays_o_valid, rays_d_valid, _, _, _ = batchified_get_rays(
                intrinsics[i_split[1]],
                extrinsics[i_split[1]],
                image_sizes,
                True,
                False,
                False,
                False,
                None,
            )
            rays_o = torch.tensor(rays_o.reshape([cps.shape[0], H, W, 3]))
            rays_d = torch.tensor(rays_d.reshape([cps.shape[0], H, W, 3]))
            rays_o_valid = torch.tensor(rays_o_valid.reshape([cps_valid.shape[0], H, W, 3]))
            rays_d_valid = torch.tensor(rays_d_valid.reshape([cps_valid.shape[0], H, W, 3]))
            # rays_o = tntdataset.rays_o.reshape([cps.shape[0], H, W, 3])
            # rays_d = tntdataset.rays_d.reshape([cps.shape[0], H, W, 3])
            # rays_o_valid = valiad_tntdataset.rays_o_valid.reshape([cps_valid.shape[0], H, W, 3])
            # rays_d_valid = valiad_tntdataset.rays_d_valid.reshape([cps_valid.shape[0], H, W, 3])

        if dataset_type == 'mip360':
            rays_o, rays_d, _, _, _ = batchified_get_rays(
                intrinsics[i_split[0]],
                extrinsics[i_split[0]],
                image_sizes,
                True,
                False,
                False,
                False,
                None,
            )
            rays_o_valid, rays_d_valid, _, _, _ = batchified_get_rays(
                intrinsics[:cps_valid.shape[0]],
                render_poses_v,
                image_sizes,
                True,
                False,
                False,
                False,
                None,
            )
            rays_o = torch.tensor(rays_o.reshape([cps.shape[0], H, W, 3]))
            rays_d = torch.tensor(rays_d.reshape([cps.shape[0], H, W, 3]))
            rays_o_valid = torch.tensor(rays_o_valid.reshape([cps_valid.shape[0], H, W, 3]))
            rays_d_valid = torch.tensor(rays_d_valid.reshape([cps_valid.shape[0], H, W, 3]))
            # rays_o = tntdataset.rays_o.reshape([cps.shape[0], H, W, 3])
            # rays_d = tntdataset.rays_d.reshape([cps.shape[0], H, W, 3])
            # rays_o_valid = valiad_tntdataset.rays_o_valid.reshape([cps_valid.shape[0], H, W, 3])
            # rays_d_valid = valiad_tntdataset.rays_d_valid.reshape([cps_valid.shape[0], H, W, 3])

        """nerf生成的内容图像的 Style Data"""
        if not os.path.exists(data_path + '/stylized_gen_' + str(factor) + '/' + '/stylized_data.npz'):
            print("Stylizing training data ...")
            style_names, style_paths, style_images, style_features, _ = style_data_prepare(style_path, images, size=512, chunk=8, sv_path=data_path + '/stylized_gen_' + str(factor) + '/', decode_path=decode_path)
            np.savez(data_path + '/stylized_gen_' + str(factor) + '/' + '/stylized_data', style_names=style_names, style_paths=style_paths, style_images=style_images, style_features=style_features)
        else:
            print("Stylized data from " + data_path + '/stylized_gen_' + str(factor) + '/' + '/stylized_data.npz')
            stylized_data = np.load(data_path + '/stylized_gen_' + str(factor) + '/' + '/stylized_data.npz', allow_pickle=True)
            style_names, style_paths, style_images, style_features = stylized_data['style_names'][()], stylized_data['style_paths'], stylized_data['style_images'], stylized_data['style_features']

        """Setting Attributes"""
        self.set_mode(mode)
        self.frame_num = cps.shape[0]
        self.h, self.w, self.f = H, W, focal
        self.hwf = hwf
        self.K = K
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.cps, self.intr, self.images = cps, K, images
        # self.imagess = imagess
        self.cps_valid = cps_valid
        self.rays_num = self.frame_num * self.h * self.w
        self.near, self.far = near, far

        self.style_names = style_names
        self.style_names_t = {y: x for x, y in self.style_names.items()}
        self.style_images = style_images
        self.style_paths = [data_path + '/stylized_gen_' + str(factor) + '/']
        self.style_features = style_features
        # self.style_img_features = style_img_features
        self.style_num = self.style_images.shape[0]

        self.store_rays = store_rays
        self.is_ndc = ((dataset_type == 'llff' or dataset_type == 'mip360') and not no_ndc)
        self.rays_o, self.rays_d = rays_o, rays_d
        self.rays_o_valid, self.rays_d_valid = rays_o_valid, rays_d_valid
        self.stylized_images_uint8 = None
        if collect_stylized_images:
            self.collect_all_stylized_images()
        print("Dataset Creation Done !")

    # 打开相应的样式化图像文件，并将其转换为np.uint8类型的RGB数组。然后，将这些图像存储到先前创建的stylized_images_uint8数组的相应位置。
    def collect_all_stylized_images(self):
        print(self.style_names.keys())
        if self.stylized_images_uint8 is not None:
            return
        self.stylized_images_uint8 = np.zeros([self.style_num, self.frame_num, self.h, self.w, 3], dtype=np.uint8)
        for i in range(self.style_num):
            print('Collecting style: ' + self.style_names_t[i])
            for j in tqdm(range(self.frame_num)):
                img = np.array(Image.open(self.style_paths[i] + '/%03d.jpg' % (j+1)).convert('RGB'), np.uint8)
                self.stylized_images_uint8[i, j] = img

    def get_item_train(self, idx):
        frame_id = idx // (self.h * self.w)
        pixel_id = idx % (self.h * self.w)
        hid, wid = pixel_id // self.w, pixel_id % self.w
        rgb = self.images[frame_id, hid, wid]
        if self.store_rays:
            ray_o = self.rays_o[frame_id, hid, wid]
            ray_d = self.rays_d[frame_id, hid, wid]
        else:
            ray_o, ray_d = get_rays_from_id(hid, wid, self.f, self.cps[frame_id], self.cx, self.cy)
            if self.is_ndc:
                ray_o, ray_d = ndc_rays_np(self.h, self.w, self.f, 1., ray_o[np.newaxis], ray_d[np.newaxis])
                ray_o, ray_d = ray_o[0], ray_d[0]
        return {'rgb_gt': rgb, 'rays_o': ray_o, 'rays_d': ray_d}

    def get_item_train_style(self, idx):
        style_id = idx // (self.frame_num * self.h * self.w)
        frame_id = (idx % (self.frame_num * self.h * self.w)) // (self.h * self.w)
        hid = (idx % (self.h * self.w)) // self.w
        wid = idx % self.w
        if self.stylized_images_uint8 is None:
            stylized_contents = np.load(self.style_paths[style_id] + '/%03d.npz' % frame_id)['stylized_image']
            rgb = stylized_contents[hid, wid]
        else:
            rgb = np.float32(self.stylized_images_uint8[style_id, frame_id, hid, wid]) / 255
        rgb_origin = self.images[frame_id, hid, wid]
        style_feature = self.style_features[style_id]
        if self.store_rays:
            ray_o = self.rays_o[frame_id, hid, wid]
            ray_d = self.rays_d[frame_id, hid, wid]
        else:
            ray_o, ray_d = get_rays_from_id(hid, wid, self.f, self.cps[frame_id], self.cx, self.cy)
            if self.is_ndc:
                ray_o, ray_d = ndc_rays_np(self.h, self.w, self.f, 1., ray_o[np.newaxis], ray_d[np.newaxis])
                ray_o, ray_d = ray_o[0], ray_d[0]
        return {'rgb_gt': rgb, 'rays_o': ray_o, 'rays_d': ray_d, 'style_feature': style_feature, 'rgb_origin': rgb_origin, 'style_id': style_id, 'frame_id': frame_id, 'hid': hid, 'wid': wid}

    def loss_coh_get_item_train_style(self, idx, style_id, frame_id):
        hid = (idx % (self.h * self.w)) // self.w
        wid = idx % self.w
        if self.stylized_images_uint8 is None:
            stylized_contents = np.load(self.style_paths[style_id] + '/%03d.npz' % frame_id)['stylized_image']
            rgb = stylized_contents[hid, wid]
        else:
            rgb = np.float32(self.stylized_images_uint8[style_id, frame_id, hid, wid]) / 255
        # rgb_origin = self.imagess[frame_id, hid, wid]
        rgb_origin = self.images[frame_id, hid, wid]
        style_feature = self.style_features[style_id]
        if self.store_rays:
            ray_o = self.rays_o[frame_id, hid, wid]
            ray_d = self.rays_d[frame_id, hid, wid]
        else:
            ray_o, ray_d = get_rays_from_id(hid, wid, self.f, self.cps[frame_id], self.cx, self.cy)
            if self.is_ndc:
                ray_o, ray_d = ndc_rays_np(self.h, self.w, self.f, 1., ray_o[np.newaxis], ray_d[np.newaxis])
                ray_o, ray_d = ray_o[0], ray_d[0]
        return {'rgb_gt': rgb, 'rays_o': ray_o, 'rays_d': ray_d, 'style_feature': style_feature, 'rgb_origin': rgb_origin, 'style_id': style_id, 'frame_id': frame_id, 'hid': hid, 'wid': wid}


    def get_item_valid(self, idx):
        frame_id = idx // (self.h * self.w)
        pixel_id = idx % (self.h * self.w)
        hid, wid = pixel_id // self.w, pixel_id % self.w
        if self.store_rays:
            ray_o = self.rays_o_valid[frame_id, hid, wid]
            ray_d = self.rays_d_valid[frame_id, hid, wid]
        else:
            ray_o, ray_d = get_rays_from_id(hid, wid, self.f, self.cps_valid[frame_id], self.cx, self.cy)
            if self.is_ndc:
                ray_o, ray_d = ndc_rays_np(self.h, self.w, self.f, 1., ray_o[np.newaxis], ray_d[np.newaxis])
                ray_o, ray_d = ray_o[0], ray_d[0]
        return {'rays_o': ray_o, 'rays_d': ray_d}

    def get_item_valid_style(self, idx):
        style_id = idx // (self.cps_valid.shape[0] * self.h * self.w)
        frame_id = (idx % (self.cps_valid.shape[0] * self.h * self.w)) // (self.h * self.w)
        hid = (idx % (self.h * self.w)) // self.w
        wid = idx % self.w
        style_feature = self.style_features[style_id]
        if self.store_rays:
            ray_o = self.rays_o_valid[frame_id, hid, wid]
            ray_d = self.rays_d_valid[frame_id, hid, wid]
        else:
            ray_o, ray_d = get_rays_from_id(hid, wid, self.f, self.cps_valid[frame_id], self.cx, self.cy)
            if self.is_ndc:
                ray_o, ray_d = ndc_rays_np(self.h, self.w, self.f, 1., ray_o[np.newaxis], ray_d[np.newaxis])
                ray_o, ray_d = ray_o[0], ray_d[0]
        return {'rays_o': ray_o, 'rays_d': ray_d, 'style_feature': style_feature, 'style_id': style_id, 'frame_id': frame_id}

    def get_patch_train(self, fid, hid, wid, patch_size=32):
        min_hid, min_wid = int(min(max(hid - patch_size / 2, 0), self.h - patch_size)), int(min(max(wid - patch_size / 2, 0), self.w - patch_size))
        max_hid, max_wid = min_hid + patch_size, min_wid + patch_size
        hids, wids = np.meshgrid(np.arange(min_hid, max_hid), np.arange(min_wid, max_wid))
        hids, wids = hids.reshape([-1]), wids.reshape([-1])
        rgbs = torch.from_numpy(np.stack([self.images[fid, hids[i], wids[i]] for i in range(hids.shape[0])], axis=0)).float()
        if self.store_rays:
            rays_o = np.stack([self.rays_o[fid, hids[i], wids[i]] for i in range(hids.shape[0])], axis=0)
            rays_d = np.stack([self.rays_d[fid, hids[i], wids[i]] for i in range(hids.shape[0])], axis=0)
        else:
            rays_od = np.stack([get_rays_from_id(hids[i], wids[i], self.f, self.cps[fid], self.cx, self.cy) for i in range(hids.shape[0])])
            rays_o, rays_d = rays_od[:, 0], rays_od[:, 1]
            if self.is_ndc:
                rays_o, rays_d = ndc_rays_np(self.h, self.w, self.f, 1., rays_o, rays_d)
        rays_o = torch.from_numpy(rays_o).float()
        rays_d = torch.from_numpy(rays_d).float()
        return {'rgb_gt': rgbs, 'ray_o': rays_o, 'rays_d': rays_d}

    def get_patch_train_style(self, style_id, fid, hid, wid, patch_size=32):
        min_hid, min_wid = int(min(max(hid - patch_size / 2, 0), self.h - patch_size)), int(min(max(wid - patch_size / 2, 0), self.w - patch_size))
        max_hid, max_wid = min_hid + patch_size, min_wid + patch_size
        hids, wids = np.meshgrid(np.arange(min_hid, max_hid), np.arange(min_wid, max_wid))
        hids, wids = hids.T.reshape([-1]), wids.T.reshape([-1])  # .T to keep the orientation of the image
        if self.stylized_images_uint8 is None:
            stylized_contents = np.load(self.style_paths[style_id] + '/%03d.npz' % fid)['stylized_image']
            rgbs = torch.from_numpy(np.stack([stylized_contents[hids[i], wids[i]] for i in range(patch_size ** 2)], axis=0)).float()
        else:
            rgbs = torch.from_numpy(np.stack([np.float32(self.stylized_images_uint8[style_id, fid, hids[i], wids[i]]) / 255 for i in range(patch_size ** 2)], axis=0)).float()
        rgbs_origin = torch.from_numpy(np.stack([self.images[fid, hids[i], wids[i]] for i in range(hids.shape[0])], axis=0)).float()
        if self.store_rays:
            rays_o = np.stack([self.rays_o[fid, hids[i], wids[i]] for i in range(hids.shape[0])], axis=0)
            rays_d = np.stack([self.rays_d[fid, hids[i], wids[i]] for i in range(hids.shape[0])], axis=0)
        else:
            rays_od = np.stack([get_rays_from_id(hids[i], wids[i], self.f, self.cps[fid], self.cx, self.cy) for i in range(hids.shape[0])])
            rays_o, rays_d = rays_od[:, 0], rays_od[:, 1]
            if self.is_ndc:
                rays_o, rays_d = ndc_rays_np(self.h, self.w, self.f, 1., rays_o, rays_d)
        rays_o = torch.from_numpy(rays_o).float()
        rays_d = torch.from_numpy(rays_d).float()
        style_image = torch.from_numpy(self.style_images[style_id:style_id+1]).float()
        style_id = torch.tensor(style_id).expand([patch_size**2]).long()
        frame_id = torch.tensor(fid).expand([patch_size**2]).long()
        return {'style_image': style_image, 'rgb_gt': rgbs, 'rays_o': rays_o, 'rays_d': rays_d, 'rgb_origin': rgbs_origin, 'style_id': style_id, 'frame_id': frame_id}

    def set_mode(self, mode='train'):
        modes = ['train', 'valid', 'train_style', 'valid_style']
        if mode not in modes:
            print('Unknown mode: ', mode, ' Only supports: ', modes)
            exit(-1)
        self.mode = mode

    def __getitem__(self, item):
        func_dict = {'train': self.get_item_train, 'valid': self.get_item_valid, 'train_style': self.get_item_train_style, 'valid_style': self.get_item_valid_style}
        return func_dict[self.mode](item)

    def getitem_forlosscoh(self, item, style_id, frame_id):
        func_dict = {'loss_coh_train_style': self.loss_coh_get_item_train_style}
        return func_dict['loss_coh_train_style'](item, style_id, frame_id)

    def __len__(self):
        if self.mode == 'train':
            return self.frame_num * self.w * self.h
        elif self.mode == 'valid':
            return self.cps_valid.shape[0] * self.w * self.h
        elif self.mode == 'train_style':
            return self.style_num * self.frame_num * self.w * self.h
        else:
            return self.style_num * self.cps_valid.shape[0] * self.w * self.h

# 用于批量加载训练数据集
class LightDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_num = len(dataset)
        self.data_idx = np.arange(self.data_num)
        self.frame_num = dataset.frame_num
        self.style_num = dataset.style_num
        if self.shuffle:
            np.random.shuffle(self.data_idx)
        self.start = 0
        self.frame_start = 0
        self.style_start = 0
        # 通过调用 dataset.__getitem__(0) 来获取数据集中的第一个样本，并获取其键值作为数据的键。
        data0 = self.dataset.__getitem__(0)
        self.keys = data0.keys()

    def get_batch(self):
        if self.batch_size >= self.data_num:
            # 从数据索引中随机选择批量大小个索引
            idx = np.random.choice(self.data_idx, self.batch_size, replace=True)
            # Initialize
            batch_data = {}
            for key in self.keys:
                batch_data[key] = []
            # Append data
            for i in range(self.batch_size):
                data = self.dataset.__getitem__(idx[i])
                for key in data.keys():
                    batch_data[key].append(data[key])
            self.start += self.batch_size
            # To tensor
            for key in self.keys:
                batch_data[key] = torch.from_numpy(np.stack(batch_data[key]))
            return batch_data

        # Check if shuffle again
        if self.start + self.batch_size >= self.data_num:
            self.start = 0
            np.random.shuffle(self.data_idx)

        # Initialize
        batch_data = {}
        for key in self.keys:
            batch_data[key] = []
        # Append data
        for i in range(self.batch_size):
            data = self.dataset.__getitem__(self.data_idx[self.start + i])
            for key in data.keys():
                batch_data[key].append(data[key])
        self.start += self.batch_size
        # To tensor
        for key in self.keys:
            batch_data[key] = torch.from_numpy(np.stack(batch_data[key])).float()
        return batch_data

    def get_batch_reverse(self):
        if self.batch_size >= self.data_num:
            # 从数据索引中随机选择批量大小个索引
            idx = np.random.choice(self.data_idx, self.batch_size, replace=True)
            # Initialize
            batch_data = {}
            for key in self.keys:
                batch_data[key] = []
            # Append data
            for i in range(self.batch_size):
                data = self.dataset.__getitem__(idx[i])
                for key in data.keys():
                    batch_data[key].append(data[key])
            self.start += self.batch_size
            # To tensor
            for key in self.keys:
                batch_data[key] = torch.from_numpy(np.stack(batch_data[key]))
            return batch_data

        # Check if shuffle again
        if self.start + self.batch_size >= self.data_num:
            self.start = 0
            np.random.shuffle(self.data_idx)

        # Initialize
        batch_data = {}
        for key in self.keys:
            batch_data[key] = []
        # Append data
        for i in range(self.batch_size):
            data = self.dataset.__getitem__(self.data_idx[self.data_num - self.start - i - 1])
            for key in data.keys():
                batch_data[key].append(data[key])
        self.start += self.batch_size
        # To tensor
        for key in self.keys:
            batch_data[key] = torch.from_numpy(np.stack(batch_data[key])).float()
        return batch_data

# 跟帧数有关，风格数可以随机，batch_size_style训练帧数的倍数
    def loss_coh_get_batch(self):
        if self.batch_size >= self.data_num:
            # 从数据索引中随机选择批量大小个索引
            idx = np.random.choice(self.data_idx, self.batch_size, replace=True)
            # Initialize
            batch_data = {}
            for key in self.keys:
                batch_data[key] = []
            # Append data
            for i in range(self.batch_size):
                data = self.dataset.__getitem__(idx[i])
                for key in data.keys():
                    batch_data[key].append(data[key])
            self.start += self.batch_size
            # To tensor
            for key in self.keys:
                batch_data[key] = torch.from_numpy(np.stack(batch_data[key]))
            return batch_data

        # Check if shuffle again
        if self.start + self.batch_size >= self.data_num:
            self.start = 0
            np.random.shuffle(self.data_idx)

        # Initialize
        batch_data = {}
        for key in self.keys:
            batch_data[key] = []
        # Append data

        for i in range(self.batch_size):
            data = self.dataset.getitem_forlosscoh(self.data_idx[self.start + i], self.style_start, self.frame_start)
            for key in data.keys():
                batch_data[key].append(data[key])
        if self.frame_start == self.frame_num - 1 and self.style_start != self.style_num - 1 and self.start >= self.dataset.h * self.dataset.w:
            self.style_start += 1
            self.frame_start = 0
            self.start = 0
        elif self.frame_start != self.frame_num - 1:
            self.frame_start += 1
        else:
            self.frame_start = 0
            self.start += self.batch_size
        # To tensor
        for key in self.keys:
            batch_data[key] = torch.from_numpy(np.stack(batch_data[key])).float()
        return batch_data