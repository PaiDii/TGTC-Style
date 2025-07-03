import torch.nn as nn
import torch
from Style_function import adaptive_instance_normalization as adain
from Style_function import calc_mean_std, styleLoss, GramMatrix
from utils import L2_norm

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(inplace=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(inplace=True),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(inplace=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(inplace=True),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(inplace=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(inplace=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()
)

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def calc_style_loss_gram(self, inputs, targets):
        style_loss = 0
        for i in range(len(inputs)):
            input, target = inputs[i], targets[i]
            style_loss += styleLoss(input, target)
        return style_loss

    def calc_nerf_loss(self, x, content_gt, style_gt):
        fea_x = self.encode_with_intermediate(x)
        fea_style_gt = self.encode_with_intermediate(style_gt)
        fea_content_gt = self.encode_with_intermediate(content_gt)
        loss_s = self.calc_style_loss_gram(fea_x, fea_style_gt)
        loss_c = self.calc_content_loss(fea_x[-1], fea_content_gt[-1])
        return loss_c, loss_s

    def forward(self, content, style, alpha=1.0, return_stylized_content=False):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        stylized_content_feat = adain(content_feat, style_feats[-1])
        stylized_content_feat = alpha * stylized_content_feat + (1 - alpha) * content_feat

        stylized_content = self.decoder(stylized_content_feat)
        stylized_content_feat_encoded = self.encode_with_intermediate(stylized_content)

        loss_c = self.calc_content_loss(stylized_content_feat_encoded[-1], stylized_content_feat)
        loss_s = self.calc_style_loss(stylized_content_feat_encoded[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(stylized_content_feat_encoded[i], style_feats[i])
        if return_stylized_content:
            return loss_c, loss_s, stylized_content, stylized_content_feat
        else:
            return loss_c, loss_s

def load_pretrained_vgg16():
    vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    vgg = nn.Sequential(*list(vgg.features.children()))
    vgg.eval()
    return vgg

def get_normalized_torch(tensor, mean, std):
    tensor = (tensor[:, [2, 1, 0]] * 255).float()
    mean = torch.tensor(mean).float().to(tensor.device)
    std = torch.tensor(std).float().to(tensor.device)
    tensor = (tensor - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    return tensor

layers = [15, 22]

def get_features(tensor):
    features = {}
    vgg = load_pretrained_vgg16().to(tensor.device)
    x = get_normalized_torch(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    for name, layer in enumerate(vgg):
        x = layer(x)
        if name in layers:
            features[name] = x
            if name == 22:
                break
    return features

def cos_loss(a, b):
    a_norm = (a * a).sum(1, keepdim=True).sqrt()
    b_norm = (b * b).sum(1, keepdim=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()

def cosine_similarity(a, b):
    a_norm = torch.norm(a, dim=1, keepdim=True)
    b_norm = torch.norm(b, dim=1, keepdim=True)
    a_normalized = a / (a_norm + 1e-8)
    b_normalized = b / (b_norm + 1e-8)
    similarity = torch.sum(a_normalized * b_normalized, dim=1)
    return similarity

def cosine_similarity_coh(a, b):
    a_norm = torch.norm(a, keepdim=True)
    b_norm = torch.norm(b, keepdim=True)
    a_normalized = a / (a_norm + 1e-8)
    b_normalized = b / (b_norm + 1e-8)
    similarity = torch.sum(a_normalized * b_normalized)
    return similarity

class loss_coh_vgg(nn.Module):
    def __init__(self, encoder):
        super(loss_coh_vgg, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:16])
        self.enc_2 = nn.Sequential(*enc_layers[16:23])
        self.mse_loss = nn.MSELoss()

        for name in ['enc_1', 'enc_2']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(2):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def cosine_similarity(self, a, b):
        a_f = self.encode_with_intermediate(a)
        b_f = self.encode_with_intermediate(b)
        similarity = []
        for i in range(2):
            a = a_f[i]
            b = b_f[i]
            a_norm = torch.norm(a, dim=1, keepdim=True)
            b_norm = torch.norm(b, dim=1, keepdim=True)
            a_normalized = a / (a_norm + 1e-8)
            b_normalized = b / (b_norm + 1e-8)
            similarity.append(torch.sum(a_normalized * b_normalized, dim=1))
        return similarity

    def forward(self, a, b, c, d):
        similarity1 = self.cosine_similarity(a, b)
        similarity2 = self.cosine_similarity(c, d)
        loss = torch.tensor(0., device=a.device)
        for i in range(2):
            loss += L2_norm(similarity1[i], similarity2[i])
        return loss