import shutil
from builtins import print

import torch.nn as nn
import VGGNet
from rendering import *
from dataset import RaySampler, StyleRaySampler, StyleRaySampler_gen, LightDataLoader
from models import StyleNerf, StyleMLP_Wild_multilayers, VAE, StyleLatents_variational, StyleMLP_before_concat
from train_style_modules import train_temporal_invoke, train_temporal_invoke_pl
from config import config_parser
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def train(args):
    """Check Nerf Type"""
    nerf_dict = {'style_nerf': StyleNerf}
    nerf_type_str = ''
    for nerf_type in nerf_dict.keys():
        nerf_type_str += (nerf_type + ' ')
    assert args.nerf_type in nerf_dict.keys(), 'Unknown nerf type: ' + args.nerf_type + '. Only support: ' + nerf_type_str
    print('Type of nerf: ', args.nerf_type)

    """Style Module Type"""
    style_module_dict = {'mlp': StyleMLP_Wild_multilayers}
    style_type_str = ''
    for style_type in style_module_dict.keys():
        style_type_str += (style_type + ' ')
    assert args.style_type in style_module_dict.keys(), 'Unknown style type: ' + args.style_type + '. Only support: ' + style_type_str
    print('Type of style: ', args.style_type)

    """concat Style Module Type"""
    concat_style_module_dict = {'mlp': StyleMLP_before_concat}
    style_type_str = ''
    for style_type in concat_style_module_dict.keys():
        style_type_str += (style_type + ' ')
    assert args.style_type in concat_style_module_dict.keys(), 'Unknown style type: ' + args.style_type + '. Only support: ' + style_type_str
    print('Type of style: ', args.style_type)

    """Latent Module Type"""
    latent_module_dict = {'variational': StyleLatents_variational}
    latent_type_str = ''
    for latent_type in latent_module_dict.keys():
        latent_type_str += (latent_type + ' ')
    assert args.latent_type in latent_module_dict.keys(), 'Unknown latent type: ' + args.latent_type + '. Only support: ' + latent_type_str
    print('Type of latent: ', args.latent_type)

    """Check Sampling Type"""
    samp_dict = {'uniform': sampling_pts_uniform}
    samp_type_str = ''
    for samp_type in samp_dict.keys():
        samp_type_str += (samp_type + ' ')
    assert args.sample_type in samp_dict.keys(), 'Unknown nerf type: ' + args.sample_type + '. Only support: ' + samp_type_str
    print('Sampling Strategy: ', args.sample_type)
    samp_func = samp_dict[args.sample_type]
    if args.N_samples_fine > 0:
        samp_func_fine = sampling_pts_fine_torch

    """Saving Configuration"""
    use_viewdir_str = '_UseViewDir_' if args.use_viewdir else ''
    sv_path = os.path.join(args.basedir, args.expname + '_' + args.nerf_type + '_' + args.act_type + use_viewdir_str + 'ImgFactor' + str(int(args.factor)))
    save_makedir(sv_path)
    shutil.copy(args.config, sv_path)
    nerf_gen_data_path = sv_path + '/nerf_gen_data2/'

    """Create Nerfs"""
    nerf = nerf_dict[args.nerf_type]
    model = nerf(args=args, mode='coarse').to(device)
    model.train()
    grad_vars = list(model.parameters())
    model_forward = batchify(lambda **kwargs: model(**kwargs), args.chunk)

    if args.N_samples_fine > 0:
        nerf_fine = nerf_dict[args.nerf_type_fine]
        model_fine = nerf_fine(args=args, mode='fine').to(device)
        model_fine.train()
        grad_vars += list(model_fine.parameters())
        model_forward_fine = batchify(lambda **kwargs: model_fine(**kwargs), args.chunk)

    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    """Create concat Style Module"""
    concat_style = concat_style_module_dict[args.style_type]
    concat_style_model = concat_style(args).to(device)
    concat_style_model.train()
    concat_style_vars = list(concat_style_model.parameters())
    concat_style_forward = batchify(lambda **kwargs: concat_style_model(**kwargs), args.chunk)

    """Create Style Module"""
    style = style_module_dict[args.style_type]
    style_model = style(args).to(device)
    style_model.train()
    style_vars = list(style_model.parameters())
    style_forward = batchify(lambda **kwargs: style_model(**kwargs), args.chunk)
    style_optimizer = torch.optim.Adam(params=style_vars + concat_style_vars, lr=args.lrate, betas=(0.9, 0.999))

    """VGG and Decoder"""
    # decoder = VGGNet.decoder
    # vgg = VGGNet.vgg
    # decoder.eval()
    # vgg.eval()
    # decoder.load_state_dict(torch.load('./pretrained/decoder.pth'))
    # vgg.load_state_dict(torch.load('./pretrained/vgg_normalised.pth'))
    # vgg = nn.Sequential(*list(vgg.children())[:31])  # relu4-1
    # vgg.to(device)
    # decoder.to(device)

    """Load Check Point"""
    global_step = 0
    ckpts_path = sv_path
    save_makedir(ckpts_path)
    ckpts = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if 'tar' in f and 'style' not in f and 'latent' not in f]
    print('Found ckpts', ckpts, ' from ', ckpts_path)

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading Nerf Model from ', ckpt_path)
        ckpt = torch.load(ckpt_path)
        print('ckpt.keys: ', ckpt.keys())
        global_step = ckpt['global_step']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if args.N_samples_fine > 0:
            model_fine.load_state_dict(ckpt['model_fine'])

    ckpts_style = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if 'tar' in f and 'style' in f and 'latent' not in f]
    if len(ckpts_style) > 0 and not args.no_reload:
        ckpt_path_style = ckpts_style[-1]
        print('Reloading Style Model from ', ckpt_path_style)
        ckpt_style = torch.load(ckpt_path_style)
        global_step = ckpt_style['global_step']
        style_model.load_state_dict(ckpt_style['model'])
        concat_style_model.load_state_dict(ckpt_style['concat_model'])
        style_optimizer.load_state_dict(ckpt_style['optimizer'])

    def Prepare_Style_data(nerf_gen_data_path):
        tmp_dataset = StyleRaySampler(data_path=args.datadir, style_path=args.styledir, factor=args.factor,
                                      valid_factor=args.gen_factor, dataset_type=args.dataset_type, mode='train',
                                      white_bkgd=args.white_bkgd, half_res=args.half_res, no_ndc=args.no_ndc,
                                      pixel_alignment=args.pixel_alignment, spherify=args.spherify, TT_far=args.TT_far)
        tmp_dataloader = DataLoader(tmp_dataset, args.batch_size_style, shuffle=False, num_workers=args.num_workers, pin_memory=(args.num_workers > 0))
        print("Preparing nerf data for style training ...")
        cal_geometry(model_forward=model_forward, samp_func=samp_func, dataloader=tmp_dataloader, args=args,
                     device=device,
                     sv_path=nerf_gen_data_path, model_forward_fine=model_forward_fine,
                     samp_func_fine=samp_func_fine)

    """Train 2D Style module"""
    if not global_step + 1 < args.origin_step:
        sv_name = 'decoder.pth'
        is_ndc = (args.dataset_type == 'llff' and not args.no_ndc)
        if not os.path.exists(sv_path + '/' + sv_name):
            if not os.path.exists(nerf_gen_data_path):
                Prepare_Style_data(nerf_gen_data_path=nerf_gen_data_path) # 生成stylized_4.0和nerf_gen_data2
            print('Training 2D Style Module')
            # 对预训练的decoder中加入三维信息以得到生成stylized_gen_4.0要用的decoder，后面这个decoder也在整体训练过程中相互学习
            if (args.dataset_type == 'llff' or args.dataset_type == 'mip360'):
                train_temporal_invoke(save_dir=sv_path, sv_name=sv_name, log_dir=sv_path + '/style_decoder/',
                                      is_ndc=is_ndc, nerf_content_dir=nerf_gen_data_path, style_dir=args.styledir, batch_size=4)
            else:
                train_temporal_invoke_pl(save_dir=sv_path, sv_name=sv_name, log_dir=sv_path + '/style_decoder/',
                                         nerf_content_dir=nerf_gen_data_path, style_dir=args.styledir, batch_size=4)
        # elif os.path.exists(sv_path + '/' + sv_name):
        #     print('Training 2D Style Module')
        #     # 对预训练的decoder中加入三维信息以得到生成stylized_gen_4.0要用的decoder，后面这个decoder也在整体训练过程中相互学习
        #     if args.dataset_type == 'llff':
        #         train_temporal_invoke(save_dir=sv_path, sv_name=sv_name, log_dir=sv_path + '/style_decoder/',
        #                               is_ndc=is_ndc,
        #                               nerf_content_dir=nerf_gen_data_path, style_dir=args.styledir, batch_size=4)
        #     else:
        #         train_temporal_invoke_pl(save_dir=sv_path, sv_name=sv_name, log_dir=sv_path + '/style_decoder/',
        #                                  nerf_content_dir=nerf_gen_data_path, style_dir=args.styledir, batch_size=4)

    """Dataset Creation"""
    if global_step + 1 < args.origin_step and not os.path.exists(nerf_gen_data_path):
        train_dataset = RaySampler(data_path=args.datadir, factor=args.factor,
                                   mode='train', valid_factor=args.valid_factor, dataset_type=args.dataset_type,
                                   white_bkgd=args.white_bkgd, half_res=args.half_res, no_ndc=args.no_ndc,
                                   pixel_alignment=args.pixel_alignment, spherify=args.spherify, TT_far=args.TT_far)
    else:
        # if not os.path.exists(nerf_gen_data_path):
        #     Prepare_Style_data(nerf_gen_data_path=nerf_gen_data_path)

        # 生成stylized_gen_4.0
        train_dataset = StyleRaySampler_gen(data_path=args.datadir, gen_path=nerf_gen_data_path,
                                            style_path=args.styledir,
                                            factor=args.factor,
                                            mode='train', valid_factor=args.valid_factor,
                                            dataset_type=args.dataset_type,
                                            white_bkgd=args.white_bkgd, half_res=args.half_res, no_ndc=args.no_ndc,
                                            pixel_alignment=args.pixel_alignment, spherify=args.spherify,
                                            decode_path=sv_path + '/decoder.pth',
                                            store_rays=args.store_rays, TT_far=args.TT_far)

        """VAE"""
        vae = VAE(data_dim=1024, latent_dim=args.vae_latent, W=args.vae_w, D=args.vae_d, kl_lambda=args.vae_kl_lambda)
        vae.eval()
        vae_ckpt = args.vae_pth_path
        vae.load_state_dict(torch.load(vae_ckpt))

        """Latents Module"""
        latent_model_class = latent_module_dict[args.latent_type]
        latents_model_1 = latent_model_class(style_num=train_dataset.style_num, frame_num=train_dataset.frame_num, latent_dim=args.vae_latent).to(device)
        vae.to(device)

        latent_ckpts = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if 'tar' in f and 'style' not in f and 'latent' in f]
        print('Found ckpts', latent_ckpts, ' from ', ckpts_path, ' For Latents Module.')

        if len(latent_ckpts) > 0 and not args.no_reload:
            latent_ckpt_path = latent_ckpts[-1]
            print('Reloading Latent Model from ', latent_ckpt_path)
            latent_ckpt = torch.load(latent_ckpt_path)
            latents_model_1.load_state_dict(latent_ckpt['train_set_1'])
        else:
            print("Initializing Latent Model")
            all_style_features = torch.from_numpy(train_dataset.style_features).float().to(device)
            # all_style_img_features = torch.from_numpy(train_dataset.style_img_features).float().to(device)
            _, style_latents_mu_1, style_latents_logvar_1 = vae.encode(all_style_features)
            # _, style_latents_mu_2, style_latents_logvar_2 = vae.encode(all_style_img_features)

            latents_model_1.style_latents_mu = torch.nn.Parameter(style_latents_mu_1.detach())
            latents_model_1.style_latents_logvar = torch.nn.Parameter(style_latents_logvar_1.detach())
            latents_model_1.set_latents()

        vae.cpu()

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(args.num_workers > 0))

    # Render valid origin map
    if args.render_valid:
        render_path = os.path.join(sv_path, 'render_valid_' + str(global_step))
        valid_dataset = train_dataset
        valid_dataset.mode = 'valid'
        valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                      pin_memory=(args.num_workers > 0))
        with torch.no_grad():
            if args.N_samples_fine > 0:
                rgb_map, t_map, rgb_map_fine, t_map_fine = render(model_forward=model_forward, samp_func=samp_func,
                                                                  dataloader=valid_dataloader,
                                                                  args=args, device=device, sv_path=render_path,
                                                                  model_forward_fine=model_forward_fine,
                                                                  samp_func_fine=samp_func_fine)
            else:
                rgb_map, t_map, _, _ = render(model_forward=model_forward, samp_func=samp_func,
                                              dataloader=valid_dataloader,
                                              args=args, device=device, sv_path=render_path)
        print('Done, saving', rgb_map.shape, t_map.shape)
        exit(0)

    # Render train origin map
    if args.render_train:
        render_path = os.path.join(sv_path, 'render_train_' + str(global_step))
        render_dataset = train_dataset
        if args.N_samples_fine > 0:
            render_train(samp_func=samp_func, model_forward=model_forward, dataset=render_dataset, args=args,
                         device=device, sv_path=render_path, model_forward_fine=model_forward_fine,
                         samp_func_fine=samp_func_fine)
        else:
            render_train(samp_func=samp_func, model_forward=model_forward, dataset=render_dataset, args=args,
                         device=device, sv_path=render_path)
        exit(0)

    # Render valid style
    if args.render_valid_style:
        render_path = os.path.join(sv_path, 'render_valid_' + str(global_step))
        model.set_enable_style(True)
        if args.N_samples_fine > 0:
            model_fine.set_enable_style(True)
        valid_dataset = train_dataset
        valid_dataset.mode = 'valid_style'
        valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                      pin_memory=(args.num_workers > 0))
        with torch.no_grad():
            if args.N_samples_fine > 0:
                _, _, rgb_map_fine, t_map_fine = render_style(model_forward=model_forward, samp_func=samp_func,
                                                              style_forward=style_forward,
                                                              concat_style_forward=concat_style_forward,
                                                              latents_model_1=latents_model_1,
                                                              dataloader=valid_dataloader, args=args, device=device,
                                                              sv_path=render_path,
                                                              model_forward_fine=model_forward_fine,
                                                              samp_func_fine=samp_func_fine,
                                                              sigma_scale=args.sigma_scale)
            else:
                rgb_map, t_map, _, _ = render_style(model_forward=model_forward, samp_func=samp_func,
                                                    style_forward=style_forward,
                                                    concat_style_forward=concat_style_forward,
                                                    latents_model_1=latents_model_1,
                                                    dataloader=valid_dataloader,
                                                    args=args, device=device, sv_path=render_path,
                                                    sigma_scale=args.sigma_scale)
        print('Done, saving', rgb_map_fine.shape, t_map_fine.shape)
        return

    # Render train style
    if args.render_train_style:
        render_path = os.path.join(sv_path, 'render_train_' + str(global_step))
        model.set_enable_style(True)
        if args.N_samples_fine > 0:
            model_fine.set_enable_style(True)
        render_dataset = train_dataset
        render_dataset.mode = 'train_style'
        if args.N_samples_fine > 0:
            render_train_style(samp_func=samp_func, model_forward=model_forward, style_forward=style_forward,
                               concat_style_forward=concat_style_forward,
                               latents_model_1=latents_model_1,
                               dataset=render_dataset, args=args, device=device, sv_path=render_path,
                               model_forward_fine=model_forward_fine,
                               samp_func_fine=samp_func_fine, sigma_scale=args.sigma_scale)
        else:
            render_train_style(samp_func=samp_func, model_forward=model_forward, style_forward=style_forward,
                               concat_style_forward=concat_style_forward,
                               latents_model_1=latents_model_1,
                               dataset=render_dataset, args=args, device=device, sv_path=render_path,
                               sigma_scale=args.sigma_scale)
        return

    # Training Loop
    def Origin_train(global_step):
        data_time, model_time, opt_time = 0, 0, 0
        fine_time = 0
        while True:
            for batch_idx, batch_data in enumerate(train_dataloader):
                for key in batch_data:
                    batch_data[key] = torch.tensor(batch_data[key].numpy()).to(device)

                rgb_gt, rays_o, rays_d = batch_data['rgb_gt'], batch_data['rays_o'], batch_data['rays_d']

                pts, ts = samp_func(rays_o=rays_o, rays_d=rays_d, N_samples=args.N_samples, near=train_dataset.near,
                                    far=train_dataset.far, perturb=True)
                ray_num, pts_num = rays_o.shape[0], args.N_samples
                rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])

                forward_t = time.time()
                ret = model_forward(pts=pts, dirs=rays_d_forward)
                pts_rgb, pts_sigma = ret['rgb'], ret['sigma']
                rgb_exp, t_exp, weights = alpha_composition(pts_rgb, pts_sigma, ts, args.sigma_noise_std)

                loss_rgb = img2mse(rgb_gt, rgb_exp)
                loss = loss_rgb

                fine_t = time.time()
                if args.N_samples_fine > 0:
                    pts_fine, ts_fine = samp_func_fine(rays_o, rays_d, ts, weights, args.N_samples_fine)
                    pts_num = args.N_samples + args.N_samples_fine
                    rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
                    ret = model_forward_fine(pts=pts_fine, dirs=rays_d_forward)
                    pts_rgb_fine, pts_sigma_fine = ret['rgb'], ret['sigma']
                    rgb_exp_fine, t_exp_fine, _ = alpha_composition(pts_rgb_fine, pts_sigma_fine, ts_fine,
                                                                    args.sigma_noise_std)
                    loss_rgb_fine = img2mse(rgb_gt, rgb_exp_fine)
                    loss = loss + loss_rgb_fine

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % args.i_print == 0:
                    psnr = mse2psnr(loss_rgb)
                    if args.N_samples_fine > 0:
                        psnr_fine = mse2psnr(loss_rgb_fine)
                        tqdm.write(
                            f"[ORIGIN TRAIN] Iter: {global_step} Loss: {loss.item()} PSNR: {psnr.item()} PSNR Fine: {psnr_fine.item()} RGB Loss: {loss_rgb.item()} RGB Fine Loss: {loss_rgb_fine.item()}"
                            f" Data time: {np.round(data_time, 2)}s Model time: {np.round(model_time, 2)}s Fine time: {np.round(fine_time, 2)}s Optimization time: {np.round(opt_time, 2)}s")
                    else:
                        tqdm.write(
                            f"[ORIGIN TRAIN] Iter: {global_step} Loss: {loss_rgb.item()} PSNR: {psnr.item()} RGB Loss: {loss_rgb.item()}"
                            f" Data time: {np.round(data_time, 2)}s Model time: {np.round(model_time, 2)}s Fine time: {np.round(fine_time, 2)}s Optimization time: {np.round(opt_time, 2)}s")

                    data_time, model_time, opt_time = 0, 0, 0
                    fine_time = 0

                decay_rate = 0.1
                decay_steps = args.lrate_decay
                new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate

                end_t = time.time()
                data_time += 0
                model_time += (fine_t - forward_t)
                fine_time += 0
                opt_time += (end_t - fine_t)

                if global_step % 500 == 0 and global_step > 0 or global_step >= args.origin_step:
                    path = os.path.join(ckpts_path, '{:06d}.tar'.format(global_step))
                    if args.N_samples_fine > 0:
                        torch.save({
                            'global_step': global_step,
                            'model': model.state_dict(),
                            'model_fine': model_fine.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'style_optimizer': style_optimizer.state_dict()
                        }, path)
                    else:
                        torch.save({
                            'global_step': global_step,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'style_optimizer': style_optimizer.state_dict()
                        }, path)
                    print('Saved checkpoints at', path)

                    ckpts = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if 'tar' in f]
                    if len(ckpts) > args.ckp_num:
                        os.remove(ckpts[0])

                global_step += 1
                if global_step > args.origin_step:
                    return global_step

    torch.autograd.set_detect_anomaly(True)
    def Style_train(global_step, train_dataset):
        data_time, model_time, opt_time = 0, 0, 0
        fine_time = 0

        """VGG Net"""
        # decoder = VGGNet.decoder
        # vgg = VGGNet.vgg
        #
        # decoder_data = torch.load(sv_path + '/decoder.pth')
        # if 'decoder' in decoder_data.keys():
        #     decoder.load_state_dict(decoder_data['decoder'])
        # else:
        #     decoder.load_state_dict(decoder_data)
        # vgg.load_state_dict(torch.load(args.vgg_pth_path))
        # vgg = nn.Sequential(*list(vgg.children())[:31])
        # style_net = VGGNet.Net(vgg, decoder)
        # style_net.to(device)

        """Dataset Mode for Style"""
        if not type(train_dataset) is StyleRaySampler_gen:
            train_dataset = StyleRaySampler_gen(data_path=args.datadir, gen_path=nerf_gen_data_path,
                                                style_path=args.styledir, factor=args.factor,
                                                mode='train', valid_factor=args.valid_factor,
                                                dataset_type=args.dataset_type,
                                                white_bkgd=args.white_bkgd, half_res=args.half_res, no_ndc=args.no_ndc,
                                                TT_far=args.TT_far,
                                                pixel_alignment=args.pixel_alignment, spherify=args.spherify,
                                                decode_path=sv_path + '/decoder.pth', store_rays=args.store_rays)
        else:
            train_dataset.collect_all_stylized_images()
        train_dataset.set_mode('train_style')
        train_dataloader = LightDataLoader(train_dataset, batch_size=args.batch_size_style, shuffle=True,
                                           num_workers=args.num_workers, pin_memory=(args.num_workers > 0))
        loss_coh_dataloader = LightDataLoader(train_dataset, batch_size=args.batch_size_style, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=(args.num_workers > 0))

        rounds_per_epoch = int(train_dataloader.data_num / train_dataloader.batch_size)
        print('DataLoader Creation Done!')

        """Model Mode for Style"""
        model.set_enable_style(True)
        model.eval()
        if args.N_samples_fine > 0:
            model_fine.set_enable_style(True)
            model_fine.eval()

        latents_model_1.set_optimizer()

        loss_c, loss_s = torch.tensor(0.), torch.tensor(0.)
        cnt = 0
        x = torch.zeros((args.batch_size_style, 3), device=device)
        y = torch.zeros((args.batch_size_style, 3), device=device)
        x_origin = torch.zeros((args.batch_size_style, 3), device=device)
        while True:
            for _ in range(rounds_per_epoch):
                batch_data = train_dataloader.get_batch()

                for key in batch_data:
                    batch_data[key] = batch_data[key].to(device)

                start_t = time.time()
                rgb_gt, rays_o, rays_d, rgb_origin = batch_data['rgb_gt'], batch_data['rays_o'], batch_data['rays_d'], batch_data['rgb_origin']
                style_id, frame_id, hid, wid = batch_data['style_id'].long().cpu(), batch_data['frame_id'].long().cpu(), batch_data['hid'].cpu(), batch_data['wid'].cpu()

                pts, ts = samp_func(rays_o=rays_o, rays_d=rays_d, N_samples=args.N_samples, near=train_dataset.near, far=train_dataset.far, perturb=True)
                ray_num, pts_num = rays_o.shape[0], args.N_samples
                rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])

                loss_coh_data = loss_coh_dataloader.loss_coh_get_batch()
                for key in loss_coh_data:
                    loss_coh_data[key] = loss_coh_data[key].to(device)
                rgb_gt2, rays_o2, rays_d2, rgb_origin2 = loss_coh_data['rgb_gt'], loss_coh_data['rays_o'], loss_coh_data['rays_d'], loss_coh_data['rgb_origin']
                style_id2, frame_id2, hid2, wid2 = loss_coh_data['style_id'].long().cpu(), loss_coh_data['frame_id'].long().cpu(), loss_coh_data['hid'].cpu(), loss_coh_data['wid'].cpu()

                pts2, ts2 = samp_func(rays_o=rays_o2, rays_d=rays_d2, N_samples=args.N_samples, near=train_dataset.near, far=train_dataset.far, perturb=True)
                ray_num2, pts_num2 = rays_o2.shape[0], args.N_samples
                rays_d_forward2 = rays_d2.unsqueeze(1).expand([ray_num2, pts_num2, 3])

                ret2 = model_forward(pts=pts2, dirs=rays_d_forward2)
                base_remap_features2, pts_sigma2, pts_embed2 = ret2['base_remap'], ret2['sigma'], ret2['pts']
                first_style_latents2 = latents_model_1(style_ids=style_id2, frame_ids=frame_id2, type=args.dataset_type)
                style_latents2 = torch.mean(first_style_latents2, dim=1, keepdims=True)
                first_style_latents_forward2 = first_style_latents2.unsqueeze(1).expand([ray_num2, pts_num2, first_style_latents2.shape[-1]])

                first_ret_style2 = concat_style_forward(x=pts_embed2, latent=first_style_latents_forward2)
                concat_features2 = first_ret_style2['concat_features']
                concated_features2 = torch.cat((base_remap_features2, concat_features2), dim=-1)
                # concated_features2 = concat_features2
                # concated_features2 = torch.cat((base_remap_features2, first_style_latents_forward2), dim=-1)

                style_latents_forward2 = torch.unsqueeze(style_latents2, dim=2).expand([ray_num2, pts_num2, first_style_latents2.shape[-1]])
                ret_style2 = style_forward(x=pts_embed2, concated=concated_features2, latent=style_latents_forward2)
                # ret_style2 = style_forward(x=pts_embed2, concated=concated_features2)

                pts_rgb_style2 = ret_style2['rgb']
                rgb_exp_style2, _, weights2 = alpha_composition(pts_rgb_style2, pts_sigma2, ts2, args.sigma_noise_std)
                loss_coh = torch.tensor(0., device=device)

                if cnt == train_dataset.frame_num:
                    x = rgb_exp_style2
                    x_origin = rgb_origin2
                else:
                    if cnt != 0:
                        loss_coh = L2_norm(VGGNet.cosine_similarity(rgb_exp_style2, x) - VGGNet.cosine_similarity(rgb_origin2, x_origin))
                    x = rgb_exp_style2
                    x_origin = rgb_origin2

                forward_t = time.time()
                ret = model_forward(pts=pts, dirs=rays_d_forward)
                base_remap_features, pts_sigma, pts_embed = ret['base_remap'], ret['sigma'], ret['pts']
                first_style_latents = latents_model_1(style_ids=style_id, frame_ids=frame_id, type=args.dataset_type)
                style_latents = torch.mean(first_style_latents, dim=1, keepdims=True)
                first_style_latents_forward = first_style_latents.unsqueeze(1).expand([ray_num, pts_num, first_style_latents.shape[-1]])

                first_ret_style = concat_style_forward(x=pts_embed, latent=first_style_latents_forward)
                concat_features = first_ret_style['concat_features']
                concated_features = torch.cat((base_remap_features, concat_features), dim=-1)
                # concated_features = concat_features
                # concated_features = torch.cat((base_remap_features, first_style_latents_forward), dim=-1)

                style_latents_forward = torch.unsqueeze(style_latents, dim=2).expand([ray_num, pts_num, first_style_latents.shape[-1]])
                print(121, style_latents.shape, style_latents_forward.shape)
                ret_style = style_forward(x=pts_embed, concated=concated_features, latent=style_latents_forward)
                # ret_style = style_forward(x=pts_embed, concated=concated_features)

                pts_rgb_style = ret_style['rgb']
                rgb_exp_style, _, weights = alpha_composition(pts_rgb_style, pts_sigma, ts, args.sigma_noise_std)

                loss_rgb = args.rgb_loss_lambda * img2mse(rgb_exp_style, rgb_gt)
                logp_loss_lambda = args.logp_loss_lambda * (args.logp_loss_decay ** int((global_step - args.origin_step) / 1000))
                loss_logp = logp_loss_lambda * latents_model_1.minus_logp(style_ids=style_id, frame_ids=frame_id)

                fine_t = time.time()
                if args.N_samples_fine > 0:
                    pts_fine2, ts_fine2 = samp_func_fine(rays_o2, rays_d2, ts2, weights2, args.N_samples_fine)
                    pts_num2 = args.N_samples + args.N_samples_fine
                    rays_d_forward2 = rays_d2.unsqueeze(1).expand([ray_num2, pts_num2, 3])
                    ret2 = model_forward_fine(pts=pts_fine2, dirs=rays_d_forward2)
                    base_remap_features_fine2, pts_sigma_fine2, pts_embed_fine2 = ret2['base_remap'], ret2['sigma'], ret2['pts']
                    first_style_latents_forward2 = first_style_latents2.unsqueeze(1).expand([ray_num2, pts_num2, first_style_latents2.shape[-1]])

                    first_ret_style_fine2 = concat_style_forward(x=pts_embed_fine2, latent=first_style_latents_forward2)
                    concat_features_fine2 = first_ret_style_fine2['concat_features']
                    concated_features_fine2 = torch.cat((base_remap_features_fine2, concat_features_fine2), dim=-1)
                    # concated_features_fine2 = concat_features_fine2
                    # concated_features_fine2 = torch.cat((base_remap_features_fine2, first_style_latents_forward2), dim=-1)

                    style_latents_forward2 = torch.unsqueeze(style_latents2, dim=2).expand([ray_num2, pts_num2, first_style_latents2.shape[-1]])
                    ret_style2 = style_forward(x=pts_embed_fine2, concated=concated_features_fine2, latent=style_latents_forward2)
                    # ret_style2 = style_forward(x=pts_embed_fine2, concated=concated_features_fine2)

                    pts_rgb_style_fine2 = ret_style2['rgb']
                    rgb_exp_style_fine2, _, _ = alpha_composition(pts_rgb_style_fine2, pts_sigma_fine2, ts_fine2, args.sigma_noise_std)

                    if cnt == train_dataset.frame_num:
                        cnt = 1
                        y = rgb_exp_style_fine2
                    else:
                        if cnt != 0:
                            # 最小化风格化图像块与内容图像块的纹理差异，以提升生成的质量与清晰度。同时计算t0帧的图像块变化与t1帧的图像块变化以此来提升帧间连续性，可以减少风格化过程中常见的闪烁（Flickering）问题
                            # 同时t0帧的图像块变化，应该与t1帧的图像块变化相匹配，以此来提升帧间连续性
                            # 画个图 very nice
                            loss_coh = loss_coh + L2_norm(VGGNet.cosine_similarity(rgb_exp_style_fine2, y) - VGGNet.cosine_similarity(rgb_origin2, x_origin))
                        y = rgb_exp_style_fine2
                        cnt += 1

                    pts_fine, ts_fine = samp_func_fine(rays_o, rays_d, ts, weights, args.N_samples_fine)
                    pts_num = args.N_samples + args.N_samples_fine
                    rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
                    ret = model_forward_fine(pts=pts_fine, dirs=rays_d_forward)
                    base_remap_features_fine, pts_sigma_fine, pts_embed_fine = ret['base_remap'], ret['sigma'], ret['pts']
                    first_style_latents_forward = first_style_latents.unsqueeze(1).expand([ray_num, pts_num, first_style_latents.shape[-1]])

                    first_ret_style_fine = concat_style_forward(x=pts_embed_fine, latent=first_style_latents_forward)
                    concat_features_fine = first_ret_style_fine['concat_features']
                    concated_features_fine = torch.cat((base_remap_features_fine, concat_features_fine), dim=-1)
                    # concated_features_fine = concat_features_fine
                    # concated_features_fine = torch.cat((base_remap_features_fine, first_style_latents_forward), dim=-1)

                    style_latents_forward = torch.unsqueeze(style_latents, dim=2).expand([ray_num, pts_num, first_style_latents.shape[-1]])
                    ret_style = style_forward(x=pts_embed_fine, concated=concated_features_fine, latent=style_latents_forward)
                    # ret_style = style_forward(x=pts_embed_fine, concated=concated_features_fine)

                    pts_rgb_style_fine = ret_style['rgb']
                    rgb_exp_style_fine, _, _ = alpha_composition(pts_rgb_style_fine, pts_sigma_fine, ts_fine, args.sigma_noise_std)
                    loss_rgb_fine = args.rgb_loss_lambda * img2mse(rgb_exp_style_fine, rgb_gt)
                    loss_rgb = loss_rgb + loss_rgb_fine

                loss = loss_rgb + loss_logp
                loss_for_style = loss_rgb + loss_logp + args.loss_coh_lambda * loss_coh

                opt_t = time.time()
                if global_step > 302000:
                    style_optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    style_optimizer.step()
                else:
                    style_optimizer.zero_grad()
                    loss_for_style.backward(retain_graph=True)
                    style_optimizer.step()

                latents_model_1.optimize(loss)

                end_t = time.time()
                data_time += (forward_t - start_t)
                model_time += (fine_t - forward_t)
                fine_time += (opt_t - fine_t)
                opt_time += (end_t - fine_t)

                if global_step > 120000 and global_step <= 122000 and global_step % 500 == 0 and global_step > 0:
                    path = os.path.join(ckpts_path, 'style_{:06d}.tar'.format(global_step))
                    torch.save({
                        'global_step': global_step,
                        'model': style_model.state_dict(),
                        'concat_model': concat_style_model.state_dict(),
                        'optimizer': style_optimizer.state_dict()
                    }, path)
                    print('Saved checkpoints at', path)

                    path = os.path.join(ckpts_path, 'latent_{:06d}.tar'.format(global_step))
                    torch.save({
                        'global_step': global_step,
                        'train_set_1': latents_model_1.state_dict()
                    }, path)
                    print('Saved checkpoints at', path)
                elif global_step > 300000 and global_step < 308001 and global_step % 1000 == 0 and global_step > 0:
                    path = os.path.join(ckpts_path, 'style_{:06d}.tar'.format(global_step))
                    torch.save({
                        'global_step': global_step,
                        'model': style_model.state_dict(),
                        'concat_model': concat_style_model.state_dict(),
                        'optimizer': style_optimizer.state_dict()
                    }, path)
                    print('Saved checkpoints at', path)

                    path = os.path.join(ckpts_path, 'latent_{:06d}.tar'.format(global_step))
                    torch.save({
                        'global_step': global_step,
                        'train_set_1': latents_model_1.state_dict()
                    }, path)
                    print('Saved checkpoints at', path)
                elif global_step % args.i_weights == 0 or global_step == args.total_step and global_step > 0:
                    path = os.path.join(ckpts_path, 'style_{:06d}.tar'.format(global_step))
                    torch.save({
                        'global_step': global_step,
                        'model': style_model.state_dict(),
                        'concat_model': concat_style_model.state_dict(),
                        'optimizer': style_optimizer.state_dict()
                    }, path)
                    print('Saved checkpoints at', path)

                    ckpts = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if
                             'tar' in f and 'style' in f and 'latent' not in f]
                    if len(ckpts) > args.ckp_num:
                        os.remove(ckpts[0])

                    path = os.path.join(ckpts_path, 'latent_{:06d}.tar'.format(global_step))
                    torch.save({
                        'global_step': global_step,
                        'train_set_1': latents_model_1.state_dict()
                    }, path)
                    print('Saved checkpoints at', path)

                    ckpts = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if
                             'tar' in f and 'style' not in f and 'latent' in f]
                    if len(ckpts) > args.ckp_num:
                        os.remove(ckpts[0])

                if global_step % args.i_print == 1:
                    tqdm.write(
                        f"[STYLE TRAIN] Iter: {global_step} Loss: {loss_for_style.item()} only Loss: {args.loss_coh_lambda * loss_coh.item()} Pixel RGB Loss: {loss_rgb.item()} -Log(p) Loss: {loss_logp.item()} Loss Content: {loss_c.item()} Loss Style: {loss_s.item()}"
                        f" Data time: {np.round(data_time, 2)}s Model time: {np.round(model_time, 2)}s Fine time: {np.round(fine_time, 2)}s Optimization time: {np.round(opt_time, 2)}s")
                    data_time, model_time, opt_time = 0, 0, 0
                    fine_time = 0

                global_step += 1
                if global_step > args.total_step:
                    return global_step

    if global_step + 1 < args.origin_step:
        print('Global Step: ', global_step, ' Origin Step: ', args.origin_step)
        print('Origin Train')
        Origin_train(global_step)
    else:
        sv_name = '/decoder.pth'
        is_ndc = ((args.dataset_type == 'llff' or args.dataset_type == 'mip360') and not args.no_ndc)
        if not os.path.exists(sv_path + sv_name):
            if (args.dataset_type == 'llff' or args.dataset_type == 'mip360'):
                train_temporal_invoke(save_dir=sv_path, sv_name=sv_name, log_dir=sv_path + '/style_decoder/',
                                      is_ndc=is_ndc,
                                      nerf_content_dir=nerf_gen_data_path, style_dir=args.styledir, batch_size=4)
            else:
                train_temporal_invoke_pl(save_dir=sv_path, sv_name=sv_name, log_dir=sv_path + '/style_decoder/',
                                         nerf_content_dir=nerf_gen_data_path, style_dir=args.styledir, batch_size=4)

        if not os.path.exists(nerf_gen_data_path):
            Prepare_Style_data(nerf_gen_data_path=nerf_gen_data_path)

        Style_train(global_step, train_dataset)
        exit(0)
    return


if __name__ == '__main__':
    args = config_parser()
    while True:
        train(args=args)