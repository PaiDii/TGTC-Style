import argparse
import os
from builtins import print
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import transformer
import tctrans
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
import cv2

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = [f for f in os.listdir(self.root) if 'depth' not in f and 'geometry' not in f]
        if os.path.isdir(os.path.join(self.root, self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)             
        else:
            self.paths = [p for p in Path(self.root).glob('*') if 'depth' not in str(p) and 'geometry' not in str(p)]
        self.transform = transform
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.paths)
    def name(self):
        return 'FlatFolderDataset'

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + 1e-5 * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 5e-4 * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_transformer(
        content_dir,
        style_dir,
        save_dir='./pretrained',
        log_dir='./trans_log',
        vgg='./pretrained/vgg_normalised.pth',
        lr=5e-4,
        lr_decay=1e-5,
        max_iter=5000,
        batch_size=8,
        style_weight=10.0,
        content_weight=7.0,
        n_threads=16,
        save_model_interval=1000,
        position_embedding='sine',
        hidden_dim=512
):
    """
    Encapsulated training function that accepts all specific configuration parameters.

    Args:
        content_dir (str): Directory path to content images.
        style_dir (str): Directory path to style images.
        vgg (str): Path to the pretrained VGG checkpoint.
        save_dir (str): Directory to save the models.
        log_dir (str): Directory to save the logs.
        lr (float): Initial learning rate.
        lr_decay (float): Learning rate decay factor.
        max_iter (int): Maximum number of training iterations.
        batch_size (int): The size of the batch.
        style_weight (float): Weight for the style loss.
        content_weight (float): Weight for the content loss.
        n_threads (int): Number of threads for the data loader.
        save_model_interval (int): Interval (in iterations) at which to save the model.
        position_embedding (str): Type of positional embedding ('sine' or 'learned').
        hidden_dim (int): The embedding dimension of the Transformer.
    """
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    vgg_encoder = tctrans.vgg
    vgg_encoder.load_state_dict(torch.load(vgg))
    vgg_encoder = nn.Sequential(*list(vgg_encoder.children())[:31])

    decoder = tctrans.decoder
    decoder.load_state_dict(torch.load('./models/decoder.pth'))
    embedding = tctrans.PatchEmbed()
    Trans = transformer.Transformer()

    network = tctrans.StyTrans(vgg_encoder, decoder, embedding, Trans)
    network.train()
    network.to(device)
    if USE_CUDA and torch.cuda.device_count() > 1:
        network = nn.DataParallel(network)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(content_dir, content_tf)
    style_dataset = FlatFolderDataset(style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=n_threads))

    optimizer = torch.optim.Adam([
        {'params': network.module.transformer.parameters()},
        {'params': network.module.embedding.parameters()},
    ], lr=lr)

    if not os.path.exists(os.path.join(save_dir, "test")):
        os.makedirs(os.path.join(save_dir, "test"))

    step = 0
    decoder_ckpts = [f for f in sorted(os.listdir(save_dir)) if 'decoder_iter_' in f]
    if decoder_ckpts:
        latest_ckpt_file = decoder_ckpts[-1]
        latest_iter = int(latest_ckpt_file.split('_')[-1].split('.')[0])

        print(f"Resuming training from iteration {latest_iter}")
        step = latest_iter

        decoder_path = os.path.join(save_dir, f'decoder_iter_{latest_iter}.pth')
        decoder.load_state_dict(torch.load(decoder_path)['decoder'])

        transformer_path = os.path.join(save_dir, f'transformer_iter_{latest_iter}.pth')
        Trans.load_state_dict(torch.load(transformer_path))

        embedding_path = os.path.join(save_dir, f'embedding_iter_{latest_iter}.pth')
        embedding.load_state_dict(torch.load(embedding_path))

    for i in tqdm(range(step, max_iter)):
        if i < 1e4:
            warmup_learning_rate(optimizer, iteration_count=i)
        else:
            adjust_learning_rate(optimizer, iteration_count=i)

        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)

        out, loss_c, loss_s, l_identity1, l_identity2 = network(content_images, style_images)

        if i % 100 == 0:
            output_name = '{:s}/test/{:s}{:s}'.format(save_dir, str(i), ".jpg")
            combined_out = torch.cat((content_images.cpu(), style_images.cpu(), out.cpu()), 0)
            save_image(combined_out, output_name, nrow=batch_size)

        loss_c = content_weight * loss_c
        loss_s = style_weight * loss_s
        loss = loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1)

        print(
            f"Iter {i}: {loss.sum().item():.4f} - Content: {loss_c.sum().item():.4f} - Style: {loss_s.sum().item():.4f} "
            f"- ID1: {l_identity1.sum().item():.4f} - ID2: {l_identity2.sum().item():.4f}")

        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
        writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
        writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
        writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
        writer.add_scalar('total_loss', loss.sum().item(), i + 1)

        if (i + 1) % save_model_interval == 0 or (i + 1) == max_iter:
            model_to_save = network.module if isinstance(network, nn.DataParallel) else network

            state_dict = model_to_save.transformer.state_dict()
            torch.save(state_dict, '{:s}/transformer_iter_{:d}.pth'.format(save_dir, i + 1))

            state_dict_decoder = model_to_save.decode.state_dict()
            sv_dict = {'decoder': state_dict_decoder, 'step': i + 1}
            torch.save(sv_dict, '{:s}/decoder_iter_{:d}.pth'.format(save_dir, i + 1))

            state_dict_embedding = model_to_save.embedding.state_dict()
            torch.save(state_dict_embedding, '{:s}/embedding_iter_{:d}.pth'.format(save_dir, i + 1))

    writer.close()


