from utils import *
import time

def render(model_forward, samp_func, dataloader, args, device, sv_path=None, model_forward_fine=None, samp_func_fine=None):
    """Render Scene into Images"""
    save_makedir(sv_path)
    dataset = dataloader.dataset
    frame_num, h, w = dataset.frame_num, dataset.h, dataset.w
    resolution = h * w
    img_id = 0
    rgb_map, t_map = None, None
    rgb_map_fine, t_map_fine = None, None
    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        # To Device as Tensor
        for key in batch_data:
            batch_data[key] = torch.tensor(batch_data[key].numpy())

        # Get data and forward
        rays_o, rays_d = batch_data['rays_o'], batch_data['rays_d']
        pts, ts = samp_func(rays_o=rays_o, rays_d=rays_d, N_samples=args.N_samples, near=dataset.near, far=dataset.far)
        ray_num, pts_num = rays_o.shape[0], args.N_samples
        rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
        ret = model_forward(pts=pts, dirs=rays_d_forward)
        pts_rgb, pts_sigma = ret['rgb'], ret['sigma']
        rgb_exp, t_exp, weights = alpha_composition(pts_rgb, pts_sigma, ts, 0)
        # Gather outputs
        # rgb_exp, t_exp = rgb_exp.detach().numpy(), t_exp.detach().numpy()
        # rgb_map = rgb_exp if rgb_map is None else np.concatenate([rgb_map, rgb_exp], axis=0)
        # t_map = t_exp if t_map is None else np.concatenate([t_map, t_exp], axis=0)

        if args.N_samples_fine > 0:
            pts_fine, ts_fine = samp_func_fine(rays_o, rays_d, ts, weights, args.N_samples_fine)
            pts_num = args.N_samples + args.N_samples_fine
            rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
            ret = model_forward_fine(pts=pts_fine, dirs=rays_d_forward)
            pts_rgb_fine, pts_sigma_fine = ret['rgb'], ret['sigma']
            rgb_exp_fine, t_exp_fine, _ = alpha_composition(pts_rgb_fine, pts_sigma_fine, ts_fine, 0)
            # Gather outputs
            rgb_exp_fine, t_exp_fine = rgb_exp_fine.detach().numpy(), t_exp_fine.detach().numpy()
            rgb_map_fine = rgb_exp_fine if rgb_map_fine is None else np.concatenate([rgb_map_fine, rgb_exp_fine], axis=0)
            t_map_fine = t_exp_fine if t_map_fine is None else np.concatenate([t_map_fine, t_exp_fine], axis=0)

        # Write to svpath
        # img_num_gathered = (rgb_map.shape[0] // resolution) - img_id
        img_num_gathered = (rgb_map_fine.shape[0] // resolution)
        if img_num_gathered > 0 and sv_path is not None:
            # sv_rgb = np.array(rgb_map[img_id * resolution: (img_id + img_num_gathered) * resolution], np.float32).reshape([img_num_gathered, h, w, 3])
            # sv_t = np.array(t_map[img_id * resolution: (img_id + img_num_gathered) * resolution], np.float32).reshape([img_num_gathered, -1])
            # sv_t = (sv_t - np.min(sv_t, axis=1, keepdims=True)) / (np.max(sv_t, axis=1, keepdims=True) - np.min(sv_t, axis=1, keepdims=True) + 1e-7)
            # sv_t = sv_t.reshape([img_num_gathered, h, w])
            # sv_rgb, sv_t = np.array(sv_rgb * 255, np.int32), np.array(sv_t * 255, np.int32)
            # for i in range(img_num_gathered):
            #     imageio.imwrite(sv_path + '/coarse_%05d.png' % (i + img_id), to8b(sv_rgb[i]))
            #     imageio.imwrite(sv_path + '/coarse_depth_%05d.png' % (i + img_id), to8b(sv_t[i]))

            if args.N_samples_fine > 0:
                print('33333333333333', img_num_gathered)
                # sv_rgb = np.array(rgb_map_fine[img_id * resolution: (img_id + img_num_gathered) * resolution],
                #                   np.float32).reshape([img_num_gathered, h, w, 3])
                sv_rgb = np.array(rgb_map_fine[: resolution], np.float32).reshape([1, h, w, 3])
                # sv_t = np.array(t_map_fine[img_id * resolution: (img_id + img_num_gathered) * resolution],
                #                 np.float32).reshape([img_num_gathered, -1])
                sv_t = np.array(t_map_fine[: resolution], np.float32).reshape([1, -1])
                sv_t = (sv_t - np.min(sv_t, axis=1, keepdims=True)) / (
                            np.max(sv_t, axis=1, keepdims=True) - np.min(sv_t, axis=1, keepdims=True) + 1e-7)
                sv_t = sv_t.reshape([img_num_gathered, h, w])
                sv_rgb, sv_t = np.array(sv_rgb * 255, np.int32), np.array(sv_t * 255, np.int32)
                # for i in range(img_num_gathered):
                #     imageio.imwrite(sv_path + '/fine_%05d.png' % (i + img_id), to8b(sv_rgb[i]))
                #     imageio.imwrite(sv_path + '/fine_depth_%05d.png' % (i + img_id), to8b(sv_t[i]))
                img_id %= frame_num + 30
                imageio.imwrite(sv_path + '/fine_%05d.png' % (img_id), to8b(sv_rgb[0]))
                imageio.imwrite(sv_path + '/fine_depth_%05d.png' % (img_id), to8b(sv_t[0]))
                if img_id == 0 or img_id % frame_num:
                    img_id += 1

            # img_id += img_num_gathered
            rgb_map_fine = rgb_map_fine[resolution:]
            t_map_fine = t_map_fine[resolution:]

    # rgb_map, t_map = np.array(rgb_map).reshape([-1, h, w, 3]), np.array(t_map).reshape([-1, h, w, 1])
    # t_map_show = np.broadcast_to(t_map, [t_map.shape[0], h, w, 3])
    # t_map_show = (t_map_show - t_map_show.min()) / (t_map_show.max() - t_map_show.min() + 1e-10)
    # if sv_path is not None:
    #     imageio.mimwrite(sv_path + '/coarse_rgb.mp4', to8b(rgb_map), fps=30, quality=8)
    #     imageio.mimwrite(sv_path + '/coarse_depth.mp4', to8b(t_map_show), fps=30, quality=8)
    #
    # if args.N_samples_fine > 0:
    #     rgb_map_fine, t_map_fine = np.array(rgb_map_fine).reshape([-1, h, w, 3]), np.array(t_map_fine).reshape([-1, h, w, 1])
    #     t_map_show = np.broadcast_to(t_map_fine, [t_map_fine.shape[0], h, w, 3])
    #     t_map_show = (t_map_show - t_map_show.min()) / (t_map_show.max() - t_map_show.min() + 1e-10)
    #     if sv_path is not None:
    #         imageio.mimwrite(sv_path + '/fine_rgb.mp4', to8b(rgb_map), fps=30, quality=8)
    #         imageio.mimwrite(sv_path + '/fine_depth.mp4', to8b(t_map_show), fps=30, quality=8)

    return rgb_map, t_map, rgb_map_fine, t_map_fine


def render_train(samp_func, model_forward, dataset, args, device, sv_path=None, model_forward_fine=None, samp_func_fine=None):
    save_makedir(sv_path)
    frame_num, h, w = dataset.frame_num, dataset.h, dataset.w

    # The largest factor of h*w closest to chunk.
    batch_size = args.chunk
    while int(h * w) % batch_size != 0:
        batch_size -= 1
    # Iteration times of each image and current iteration
    iter_img = int(h * w / batch_size)
    iter = 0
    img_count = 0
    print('Pick batch size: ', batch_size)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=args.num_workers, pin_memory=(args.num_workers > 0))

    pred_rgb, gt_rgb, pred_t = [], [], []
    pred_rgb_fine, pred_t_fine = [], []
    # Iteration
    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        # To Device as Tensor
        for key in batch_data:
            batch_data[key] = torch.tensor(batch_data[key].numpy())

        # Get batch data
        rgb_gt, rays_o, rays_d = batch_data['rgb_gt'], batch_data['rays_o'], batch_data['rays_d']
        pts, ts = samp_func(rays_o=rays_o, rays_d=rays_d, N_samples=args.N_samples, near=dataset.near, far=dataset.far)
        ray_num, pts_num = rays_o.shape[0], args.N_samples
        rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])

        # Forward and Composition
        ret = model_forward(pts=pts, dirs=rays_d_forward)
        pts_rgb, pts_sigma = ret['rgb'], ret['sigma']
        rgb_exp, t_exp, weights = alpha_composition(pts_rgb, pts_sigma, ts, 0)

        pred_rgb.append(rgb_exp.detach().numpy())
        pred_t.append(t_exp.detach().numpy())
        gt_rgb.append(rgb_gt.detach().numpy())

        if args.N_samples_fine > 0:
            pts_fine, ts_fine = samp_func_fine(rays_o, rays_d, ts, weights, args.N_samples_fine)
            pts_num = args.N_samples + args.N_samples_fine
            rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
            ret = model_forward_fine(pts=pts_fine, dirs=rays_d_forward)
            pts_rgb_fine, pts_sigma_fine = ret['rgb'], ret['sigma']
            rgb_exp_fine, t_exp_fine, _ = alpha_composition(pts_rgb_fine, pts_sigma_fine, ts_fine, 0)
            pred_rgb_fine.append(rgb_exp_fine.detach().numpy())
            pred_t_fine.append(t_exp_fine.detach().numpy())

        iter += 1

        # Complete per image
        if iter == iter_img:
            # Reshape
            pred_rgb = np.concatenate(pred_rgb, axis=0).reshape([h, w, 3])
            pred_t = np.concatenate(pred_t, axis=0).reshape([h, w])
            gt_rgb = np.concatenate(gt_rgb, axis=0).reshape([h, w, 3])

            # Broadcast
            pred_t = np.broadcast_to(pred_t[..., np.newaxis], [h, w, 3])
            # Normalize
            pred_t = (pred_t - np.min(pred_t)) / (np.max(pred_t) - np.min(pred_t))

            # To 255
            pred_rgb, pred_t = np.array(pred_rgb * 255, np.int32), np.array(pred_t * 255, np.int32)
            gt_rgb = np.array(gt_rgb * 255, np.int32)

            # Saving images
            imageio.imwrite(sv_path + '/coarse_%05d.png' % img_count, to8b(pred_rgb))
            imageio.imwrite(sv_path + '/coarse_depth_%05d.png' % img_count, to8b(pred_t))
            imageio.imwrite(sv_path + '/gt_%05d.png' % img_count, to8b(gt_rgb))

            if args.N_samples_fine > 0:
                pred_rgb_fine = np.concatenate(pred_rgb_fine, axis=0).reshape([h, w, 3])
                pred_t_fine = np.concatenate(pred_t_fine, axis=0).reshape([h, w])
                pred_t_fine = np.broadcast_to(pred_t_fine[..., np.newaxis], [h, w, 3])
                pred_t_fine = (pred_t_fine - pred_t_fine.min()) / (pred_t_fine.max() - pred_t_fine.min())
                pred_rgb_fine, pred_t_fine = np.array(pred_rgb_fine * 255, np.int32), np.array(pred_t_fine * 255, np.int32)
                imageio.imwrite(sv_path + '/fine_%05d.png' % img_count, to8b(pred_rgb_fine))
                imageio.imwrite(sv_path + '/fine_depth_%05d.png' % img_count, to8b(pred_t_fine))

            img_count += 1
            print("Finish %d Image ..." % img_count)
            iter = 0
            pred_rgb, gt_rgb, pred_t, gt_t, depth_masks = [], [], [], [], []
            pred_rgb_fine, pred_t_fine = [], []

# function: 从chunk个射线的所有采样点的颜色和密度获得每个射线的rgb，深度，从而获得不同视角的rgb图和深度图并保存
# input: model_forward: func, samp_func: func, dataloader, args, device, sv_path, model_forward_fine, samp_func_fine
# output: rgb_map[20, 756, 1008, 3], t_map[20, 756, 1008, 1]
def cal_geometry(model_forward, samp_func, dataloader, args, device, sv_path=None, model_forward_fine=None, samp_func_fine=None):
    """Render Scene into Images"""
    save_makedir(sv_path)
    dataset = dataloader.dataset
    cps = dataset.cps if 'train' in dataset.mode else dataset.cps_valid
    hwf = dataset.hwf
    near, far = dataset.near, dataset.far
    frame_num, h, w = dataset.frame_num if 'train' in dataset.mode else dataset.cps_valid.shape[0], dataset.h, dataset.w
    resolution = h * w
    img_id, pixel_id = 0, 0
    # rgb_map[20*756*1008, 3], t_map深度图[20*756*1008]
    rgb_map, t_map = np.zeros([frame_num*h*w, 3], dtype=np.float32), np.zeros([frame_num*h*w], dtype=np.float32)
    # coor_map[20*756*1008, 3]
    coor_map = np.zeros([frame_num*h*w, 3], dtype=np.float32)
    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        # To Device as Tensor
        for key in batch_data:
            batch_data[key] = torch.tensor(batch_data[key].numpy()).to(device)

        # Get data and forward
        # rays_o, rays_d: [1024, 3], [1024, 3]
        rays_o, rays_d = batch_data['rays_o'], batch_data['rays_d']
        # pts: [1024, 64, 3], ts: [1024, 64]
        pts, ts = samp_func(rays_o=rays_o, rays_d=rays_d, N_samples=args.N_samples, near=dataset.near, far=dataset.far)
        # ray_num = 1024, pts_num = 64
        ray_num, pts_num = rays_o.shape[0], args.N_samples
        rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])  # [1024, 64, 3]
        ret = model_forward(pts=pts, dirs=rays_d_forward)
        # pts_rgb: [1024, 64, 3]
        # pts_sigma: [1024, 64]
        pts_rgb, pts_sigma = ret['rgb'], ret['sigma']
        # rgb_exp: [1024, 3], t_exp：[1024]， weights：[1024, 64]
        rgb_exp, t_exp, weights = alpha_composition(pts_rgb, pts_sigma, ts, 0)
        # Gather outputs
        if not args.N_samples > 0:
            rgb_exp_tmp, t_exp_tmp = rgb_exp.detach().numpy(), t_exp.detach().numpy()
            # ndarray[..., np.newaxis]：在最后增加一个维度，numpy的方法，像torch的.unsqueeze(-1)
            coor_tmp = t_exp_tmp[..., np.newaxis] * rays_d.numpy() + rays_o.numpy()
        else:
            # pts_fine: [1024, 128, 3], ts_fine: [1024, 128]
            pts_fine, ts_fine = samp_func_fine(rays_o, rays_d, ts, weights, args.N_samples_fine)
            pts_num = args.N_samples + args.N_samples_fine  # 128
            rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
            ret = model_forward_fine(pts=pts_fine, dirs=rays_d_forward)
            # pts_rgb_fine: [1024, 128, 3]
            # pts_sigma_fine: [1024, 128]
            pts_rgb_fine, pts_sigma_fine = ret['rgb'], ret['sigma']
            # rgb_exp_fine: [1024, 3], t_exp_fine：[1024]
            rgb_exp_fine, t_exp_fine, _ = alpha_composition(pts_rgb_fine, pts_sigma_fine, ts_fine, 0)
            # Gather outputs
            rgb_exp_tmp, t_exp_tmp = rgb_exp_fine.detach().cpu().numpy(), t_exp_fine.detach().cpu().numpy()
            coor_tmp = t_exp_tmp[..., np.newaxis] * rays_d.detach().cpu().numpy() + rays_o.detach().cpu().numpy()

        batch_size = coor_tmp.shape[0]  # 1024
        rgb_map[pixel_id: pixel_id+batch_size] = rgb_exp_tmp
        t_map[pixel_id: pixel_id+batch_size] = t_exp_tmp
        coor_map[pixel_id: pixel_id+batch_size] = coor_tmp
        pixel_id += batch_size

        # Write to sv_path, save in folder: nerf_gen_data2
        # img_id total 20
        img_num_gathered = (pixel_id // resolution) - img_id
        if img_num_gathered > 0 and sv_path is not None:
            sv_rgb = np.array(rgb_map[img_id * resolution: (img_id + img_num_gathered) * resolution], np.float32).reshape([img_num_gathered, h, w, 3])
            sv_t = np.array(t_map[img_id * resolution: (img_id + img_num_gathered) * resolution], np.float32).reshape([img_num_gathered, -1])
            sv_coor_map = np.array(coor_map[img_id * resolution: (img_id + img_num_gathered) * resolution], np.float32).reshape([img_num_gathered, h, w, 3])
            sv_t = (sv_t - np.min(sv_t, axis=1, keepdims=True)) / (np.max(sv_t, axis=1, keepdims=True) - np.min(sv_t, axis=1, keepdims=True) + 1e-7)
            sv_t = sv_t.reshape([img_num_gathered, h, w])
            sv_rgb, sv_t = np.array(sv_rgb * 255, np.int32), np.array(sv_t * 255, np.int32)
            for i in range(img_num_gathered):
                imageio.imwrite(sv_path + '/rgb_%05d.png' % (i + img_id), to8b(sv_rgb[i]))
                imageio.imwrite(sv_path + '/depth_%05d.png' % (i + img_id), to8b(sv_t[i]))
                np.savez(sv_path + '/geometry_%05d' % (i + img_id), coor_map=sv_coor_map[i], cps=cps[i + img_id], hwf=hwf, near=near, far=far)
            img_id += img_num_gathered

    rgb_map, t_map = np.array(rgb_map).reshape([-1, h, w, 3]), np.array(t_map).reshape([-1, h, w, 1])
    coor_map = np.array(coor_map).reshape([-1, h, w, 3])

    # geometry.npz中包含coor_map空间点坐标, cps相机位姿, hwf, near, far
    np.savez(sv_path + '/geometry', coor_map=coor_map, cps=cps, hwf=hwf, near=near, far=far)

    t_map_show = np.broadcast_to(t_map, [t_map.shape[0], h, w, 3])
    t_map_show = (t_map_show - t_map_show.min()) / (t_map_show.max() - t_map_show.min() + 1e-10)
    # if sv_path is not None:
        # imageio.mimwrite(sv_path + '/rgb.mp4', to8b(rgb_map), fps=30, quality=8)
        # imageio.mimwrite(sv_path + '/depth.mp4', to8b(t_map_show), fps=30, quality=8)

    # rgb_map[20, 756, 1008, 3], t_map[20, 756, 1008, 1]
    return rgb_map, t_map

def render_style_reverse(model_forward, samp_func, style_forward, concat_style_forward, latents_model_1,
                 dataloader, args, device, sv_path=None, model_forward_fine=None, samp_func_fine=None, sigma_scale=0.):
    """Render Scene into Images"""
    latents_model_1.rescale_sigma(sigma_scale=sigma_scale)
    # latents_model_2.rescale_sigma(sigma_scale=sigma_scale)
    save_makedir(sv_path)
    dataset = dataloader.dataset
    dataset.mode = 'valid_style'
    # frame_num = 60
    frame_num, h, w = dataset.cps_valid.shape[0], dataset.h, dataset.w
    resolution = h * w
    img_id = 0
    image_id = 0
    style_idx = 0
    # rgb_map, t_map = None, None
    rgb_map_fine, t_map_fine = None, None
    rounds_per_epoch = int(dataloader.data_num / dataloader.batch_size)
    for _ in tqdm(range(rounds_per_epoch)):
        batch_data = dataloader.get_batch_reverse()
        # To Device as Tensor
        for key in batch_data:
            batch_data[key] = torch.tensor(batch_data[key].numpy())

        # Get data and forward
        rays_o, rays_d, style_feature = batch_data['rays_o'], batch_data['rays_d'], batch_data['style_feature']
        style_id, frame_id = batch_data['style_id'].long(), batch_data['frame_id'].long()
        # Sample
        pts, ts = samp_func(rays_o=rays_o, rays_d=rays_d, N_samples=args.N_samples, near=dataset.near, far=dataset.far, perturb=True)
        ray_num, pts_num = rays_o.shape[0], args.N_samples
        rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
        # Forward
        ret = model_forward(pts=pts, dirs=rays_d_forward)
        base_remap_features, pts_sigma, pts_embed = ret['base_remap'], ret['sigma'], ret['pts']
        # style module 1
        first_style_latents = latents_model_1(style_ids=style_id, frame_ids=frame_id)
        style_latents = torch.mean(first_style_latents, dim=1, keepdims=True)
        first_style_latents_forward = first_style_latents.unsqueeze(1).expand([ray_num, pts_num, first_style_latents.shape[-1]])
        first_ret_style = concat_style_forward(x=pts_embed, latent=first_style_latents_forward)
        concat_features = first_ret_style['concat_features']
        # concated_features: [1024, 64, 512]
        concated_features = torch.concat((base_remap_features, concat_features), dim=-1)
        # Stylize
        # style_latents = latents_model_2(style_ids=style_id, frame_ids=frame_id)
        # style_latents_forward = style_latents.unsqueeze(1).expand([ray_num, pts_num, style_latents.shape[-1]])
        style_latents_forward = torch.unsqueeze(style_latents, dim=2).expand([ray_num, pts_num, first_style_latents.shape[-1]])
        ret_style = style_forward(x=pts_embed, concated=concated_features, latent=style_latents_forward)
        pts_rgb_style = ret_style['rgb']
        # Composition
        rgb_exp_style, t_exp, weights = alpha_composition(pts_rgb_style, pts_sigma, ts, 0)

        # Gather outputs
        # rgb_exp, t_exp = rgb_exp_style.detach().numpy(), t_exp.detach().numpy()
        # rgb_map = rgb_exp if rgb_map is None else np.concatenate([rgb_map, rgb_exp], axis=0)
        # t_map = t_exp if t_map is None else np.concatenate([t_map, t_exp], axis=0)

        if args.N_samples_fine > 0:
            # Sample
            pts_fine, ts_fine = samp_func_fine(rays_o, rays_d, ts, weights, args.N_samples_fine)
            pts_num = args.N_samples + args.N_samples_fine
            rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
            # Forward
            ret = model_forward_fine(pts=pts_fine, dirs=rays_d_forward)
            base_remap_features_fine, pts_sigma_fine, pts_embed_fine = ret['base_remap'], ret['sigma'], ret['pts']
            # style module 1
            first_style_latents_forward = first_style_latents.unsqueeze(1).expand(
                [ray_num, pts_num, first_style_latents.shape[-1]])
            first_ret_style_fine = concat_style_forward(x=pts_embed_fine, latent=first_style_latents_forward)
            concat_features_fine = first_ret_style_fine['concat_features']
            # concated_features_fine: [1024, 128, 512]
            concated_features_fine = torch.concat((base_remap_features_fine, concat_features_fine), dim=-1)
            # Stylize
            # style_latents_forward = style_latents.unsqueeze(1).expand([ray_num, pts_num, style_latents.shape[-1]])
            style_latents_forward = torch.unsqueeze(style_latents, dim=2).expand([ray_num, pts_num, first_style_latents.shape[-1]])
            ret_style = style_forward(x=pts_embed_fine, concated=concated_features_fine, latent=style_latents_forward)
            pts_rgb_style_fine = ret_style['rgb']
            # Composition
            rgb_exp_style_fine, t_exp_fine, _ = alpha_composition(pts_rgb_style_fine, pts_sigma_fine, ts_fine, 0)
            # Gather outputs
            rgb_exp_fine, t_exp_fine = rgb_exp_style_fine.detach().numpy(), t_exp_fine.detach().numpy()
            rgb_map_fine = rgb_exp_fine if rgb_map_fine is None else np.concatenate([rgb_map_fine, rgb_exp_fine], axis=0)
            t_map_fine = t_exp_fine if t_map_fine is None else np.concatenate([t_map_fine, t_exp_fine], axis=0)

        # Write to svpath
        # img_num_gathered = (rgb_map_fine.shape[0] // resolution) - img_id
        img_num_gathered = (rgb_map_fine.shape[0] // resolution)
        if img_num_gathered > 0 and sv_path is not None:
            # sv_rgb = np.array(rgb_map[img_id * resolution: (img_id + img_num_gathered) * resolution], np.float32).reshape([img_num_gathered, h, w, 3])
            # sv_t = np.array(t_map[img_id * resolution: (img_id + img_num_gathered) * resolution], np.float32).reshape([img_num_gathered, -1])
            # sv_t = (sv_t - np.min(sv_t, axis=1, keepdims=True)) / (np.max(sv_t, axis=1, keepdims=True) - np.min(sv_t, axis=1, keepdims=True) + 1e-7)
            # sv_t = sv_t.reshape([img_num_gathered, h, w])
            # sv_rgb, sv_t = np.array(sv_rgb * 255, np.int32), np.array(sv_t * 255, np.int32)
            # for i in range(img_num_gathered):
            #     style_id = (i + img_id) // frame_num
            #     image_id = (i + img_id) % frame_num
            #     imageio.imwrite(sv_path + '/style_%05d_coarse_%05d.png' % (style_id, image_id), to8b(sv_rgb[i]))
            #     imageio.imwrite(sv_path + '/style_%05d_coarse_depth_%05d.png' % (style_id, image_id), to8b(sv_t[i]))

            if args.N_samples_fine > 0:
                print('33333333333333', img_num_gathered)
                # sv_rgb = np.array(rgb_map_fine[img_id * resolution: (img_id + img_num_gathered) * resolution], np.float32).reshape([img_num_gathered, h, w, 3])
                sv_rgb = np.array(rgb_map_fine[: resolution], np.float32).reshape([1, h, w, 3])
                # sv_t = np.array(t_map_fine[img_id * resolution: (img_id + img_num_gathered) * resolution], np.float32).reshape([img_num_gathered, -1])
                sv_t = np.array(t_map_fine[: resolution], np.float32).reshape([1, -1])
                sv_t = (sv_t - np.min(sv_t, axis=1, keepdims=True)) / ( np.max(sv_t, axis=1, keepdims=True) - np.min(sv_t, axis=1, keepdims=True) + 1e-7)
                sv_t = sv_t.reshape([img_num_gathered, h, w])
                sv_rgb, sv_t = np.array(sv_rgb * 255, np.int32), np.array(sv_t * 255, np.int32)
                # for i in range(img_num_gathered):
                    # style_id = (i + img_id) // frame_num
                    # image_id = (i + img_id) % frame_num
                    # imageio.imwrite(sv_path + '/style_%05d_fine_%05d.png' % (style_id, image_id), to8b(sv_rgb[i]))
                    # imageio.imwrite(sv_path + '/style_%05d_fine_depth_%05d.png' % (style_id, image_id), to8b(sv_t[i]))
                if image_id == frame_num:
                    style_idx += 1
                image_id %= frame_num
                imageio.imwrite(sv_path + '/style_%05d_fine_%05d.png' % (style_idx, image_id), to8b(sv_rgb[0]))
                imageio.imwrite(sv_path + '/style_%05d_fine_depth_%05d.png' % (style_idx, image_id), to8b(sv_t[0]))
                if image_id == 0 or image_id % frame_num:
                    image_id += 1
            # img_id += img_num_gathered
            rgb_map_fine = rgb_map_fine[resolution:]
            t_map_fine = t_map_fine[resolution:]
    # rgb_map, t_map = np.array(rgb_map).reshape([-1, h, w, 3]), np.array(t_map).reshape([-1, h, w, 1])
    # t_map_show = np.broadcast_to(t_map, [t_map.shape[0], h, w, 3])
    # t_map_show = (t_map_show - t_map_show.min()) / (t_map_show.max() - t_map_show.min() + 1e-10)
    # if sv_path is not None:
        # imageio.mimwrite(sv_path + '/coarse_rgb.mp4', to8b(rgb_map), fps=30, quality=8)
        # imageio.mimwrite(sv_path + '/coarse_depth.mp4', to8b(t_map_show), fps=30, quality=8)

    # if args.N_samples_fine > 0:
        # rgb_map_fine, t_map_fine = np.array(rgb_map_fine).reshape([-1, h, w, 3]), np.array(t_map_fine).reshape([-1, h, w, 1])
        # t_map_show = np.broadcast_to(t_map_fine, [t_map_fine.shape[0], h, w, 3])
        # t_map_show = (t_map_show - t_map_show.min()) / (t_map_show.max() - t_map_show.min() + 1e-10)
        # if sv_path is not None:
            # imageio.mimwrite(sv_path + '/fine_rgb.mp4', to8b(rgb_map), fps=30, quality=8)
            # imageio.mimwrite(sv_path + '/fine_depth.mp4', to8b(t_map_show), fps=30, quality=8)

    # return rgb_map, t_map, rgb_map_fine, t_map_fine
    return rgb_map_fine, t_map_fine


def render_style(model_forward, samp_func, style_forward, concat_style_forward, latents_model_1,
                 dataloader, args, device, sv_path=None, model_forward_fine=None, samp_func_fine=None, sigma_scale=0.):
    """Render Scene into Images"""
    latents_model_1.sigma_scale = sigma_scale
    # latents_model_2.rescale_sigma(sigma_scale=sigma_scale)
    save_makedir(sv_path)
    dataset = dataloader.dataset
    dataset.mode = 'valid_style'
    # frame_num = 60
    frame_num, h, w = dataset.cps_valid.shape[0], dataset.h, dataset.w
    resolution = h * w
    img_id = 0
    image_id = 0
    style_idx = 0
    # rgb_map, t_map = None, None
    rgb_map_fine, t_map_fine = None, None
    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        # To Device as Tensor
        for key in batch_data:
            batch_data[key] = torch.tensor(batch_data[key].numpy()).to(device)

        # Get data and forward
        rays_o, rays_d, style_feature = batch_data['rays_o'], batch_data['rays_d'], batch_data['style_feature']
        style_id, frame_id = batch_data['style_id'].long().detach().cpu(), batch_data['frame_id'].long().detach().cpu()
        # Sample
        pts, ts = samp_func(rays_o=rays_o, rays_d=rays_d, N_samples=args.N_samples, near=dataset.near, far=dataset.far, perturb=True)
        ray_num, pts_num = rays_o.shape[0], args.N_samples
        rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
        # Forward
        ret = model_forward(pts=pts, dirs=rays_d_forward)
        base_remap_features, pts_sigma, pts_embed = ret['base_remap'], ret['sigma'], ret['pts']
        # style module 1
        first_style_latents = latents_model_1(style_ids=style_id, frame_ids=frame_id, type=args.dataset_type)
        style_latents = torch.mean(first_style_latents, dim=1, keepdims=True)
        first_style_latents_forward = first_style_latents.unsqueeze(1).expand([ray_num, pts_num, first_style_latents.shape[-1]])

        first_ret_style = concat_style_forward(x=pts_embed, latent=first_style_latents_forward)
        concat_features = first_ret_style['concat_features']
        # concated_features: [1024, 64, 512]
        concated_features = torch.concat((base_remap_features, concat_features), dim=-1)
        # concated_features = concat_features
        # concated_features = torch.concat((base_remap_features, first_style_latents_forward), dim=-1)

        # Stylize
        # style_latents = latents_model_2(style_ids=style_id, frame_ids=frame_id)
        # style_latents_forward = style_latents.unsqueeze(1).expand([ray_num, pts_num, style_latents.shape[-1]])
        style_latents_forward = torch.unsqueeze(style_latents, dim=2).expand([ray_num, pts_num, first_style_latents.shape[-1]])
        ret_style = style_forward(x=pts_embed, concated=concated_features, latent=style_latents_forward)
        # ret_style = style_forward(x=pts_embed, concated=concated_features)

        pts_rgb_style = ret_style['rgb']
        # Composition
        rgb_exp_style, t_exp, weights = alpha_composition(pts_rgb_style, pts_sigma, ts, 0)

        # Gather outputs
        # rgb_exp, t_exp = rgb_exp_style.detach().numpy(), t_exp.detach().numpy()
        # rgb_map = rgb_exp if rgb_map is None else np.concatenate([rgb_map, rgb_exp], axis=0)
        # t_map = t_exp if t_map is None else np.concatenate([t_map, t_exp], axis=0)

        if args.N_samples_fine > 0:
            # Sample
            pts_fine, ts_fine = samp_func_fine(rays_o, rays_d, ts, weights, args.N_samples_fine)
            pts_num = args.N_samples + args.N_samples_fine
            rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
            # Forward
            ret = model_forward_fine(pts=pts_fine, dirs=rays_d_forward)
            base_remap_features_fine, pts_sigma_fine, pts_embed_fine = ret['base_remap'], ret['sigma'], ret['pts']
            # style module 1
            first_style_latents_forward = first_style_latents.unsqueeze(1).expand([ray_num, pts_num, first_style_latents.shape[-1]])

            first_ret_style_fine = concat_style_forward(x=pts_embed_fine, latent=first_style_latents_forward)
            concat_features_fine = first_ret_style_fine['concat_features']
            # concated_features_fine: [1024, 128, 512]
            concated_features_fine = torch.concat((base_remap_features_fine, concat_features_fine), dim=-1)
            # concated_features_fine = concat_features_fine
            # concated_features_fine = torch.concat((base_remap_features_fine, first_style_latents_forward), dim=-1)

            # Stylize
            # style_latents_forward = style_latents.unsqueeze(1).expand([ray_num, pts_num, style_latents.shape[-1]])
            style_latents_forward = torch.unsqueeze(style_latents, dim=2).expand([ray_num, pts_num, first_style_latents.shape[-1]])
            ret_style = style_forward(x=pts_embed_fine, concated=concated_features_fine, latent=style_latents_forward)
            # ret_style = style_forward(x=pts_embed_fine, concated=concated_features_fine)

            pts_rgb_style_fine = ret_style['rgb']
            # Composition
            rgb_exp_style_fine, t_exp_fine, _ = alpha_composition(pts_rgb_style_fine, pts_sigma_fine, ts_fine, 0)
            # Gather outputs
            rgb_exp_fine, t_exp_fine = rgb_exp_style_fine.detach().cpu().numpy(), t_exp_fine.detach().cpu().numpy()
            rgb_map_fine = rgb_exp_fine if rgb_map_fine is None else np.concatenate([rgb_map_fine, rgb_exp_fine], axis=0)
            t_map_fine = t_exp_fine if t_map_fine is None else np.concatenate([t_map_fine, t_exp_fine], axis=0)

        # Write to svpath
        # img_num_gathered = (rgb_map_fine.shape[0] // resolution) - img_id
        img_num_gathered = (rgb_map_fine.shape[0] // resolution)
        if img_num_gathered > 0 and sv_path is not None:
            # sv_rgb = np.array(rgb_map[img_id * resolution: (img_id + img_num_gathered) * resolution], np.float32).reshape([img_num_gathered, h, w, 3])
            # sv_t = np.array(t_map[img_id * resolution: (img_id + img_num_gathered) * resolution], np.float32).reshape([img_num_gathered, -1])
            # sv_t = (sv_t - np.min(sv_t, axis=1, keepdims=True)) / (np.max(sv_t, axis=1, keepdims=True) - np.min(sv_t, axis=1, keepdims=True) + 1e-7)
            # sv_t = sv_t.reshape([img_num_gathered, h, w])
            # sv_rgb, sv_t = np.array(sv_rgb * 255, np.int32), np.array(sv_t * 255, np.int32)
            # for i in range(img_num_gathered):
            #     style_id = (i + img_id) // frame_num
            #     image_id = (i + img_id) % frame_num
            #     imageio.imwrite(sv_path + '/style_%05d_coarse_%05d.png' % (style_id, image_id), to8b(sv_rgb[i]))
            #     imageio.imwrite(sv_path + '/style_%05d_coarse_depth_%05d.png' % (style_id, image_id), to8b(sv_t[i]))

            if args.N_samples_fine > 0:
                print('33333333333333', img_num_gathered)
                # sv_rgb = np.array(rgb_map_fine[img_id * resolution: (img_id + img_num_gathered) * resolution], np.float32).reshape([img_num_gathered, h, w, 3])
                sv_rgb = np.array(rgb_map_fine[: resolution], np.float32).reshape([1, h, w, 3])
                # sv_t = np.array(t_map_fine[img_id * resolution: (img_id + img_num_gathered) * resolution], np.float32).reshape([img_num_gathered, -1])
                sv_t = np.array(t_map_fine[: resolution], np.float32).reshape([1, -1])
                sv_t = (sv_t - np.min(sv_t, axis=1, keepdims=True)) / ( np.max(sv_t, axis=1, keepdims=True) - np.min(sv_t, axis=1, keepdims=True) + 1e-7)
                sv_t = sv_t.reshape([img_num_gathered, h, w])
                sv_rgb, sv_t = np.array(sv_rgb * 255, np.int32), np.array(sv_t * 255, np.int32)
                # for i in range(img_num_gathered):
                    # style_id = (i + img_id) // frame_num
                    # image_id = (i + img_id) % frame_num
                    # imageio.imwrite(sv_path + '/style_%05d_fine_%05d.png' % (style_id, image_id), to8b(sv_rgb[i]))
                    # imageio.imwrite(sv_path + '/style_%05d_fine_depth_%05d.png' % (style_id, image_id), to8b(sv_t[i]))
                if image_id == frame_num:
                    style_idx += 1
                image_id %= frame_num
                imageio.imwrite(sv_path + '/style_%05d_fine_%05d.png' % (style_idx, image_id), to8b(sv_rgb[0]))
                imageio.imwrite(sv_path + '/style_%05d_fine_depth_%05d.png' % (style_idx, image_id), to8b(sv_t[0]))
                if image_id == 0 or image_id % frame_num:
                    image_id += 1
            # img_id += img_num_gathered
            rgb_map_fine = rgb_map_fine[resolution:]
            t_map_fine = t_map_fine[resolution:]
    # rgb_map, t_map = np.array(rgb_map).reshape([-1, h, w, 3]), np.array(t_map).reshape([-1, h, w, 1])
    # t_map_show = np.broadcast_to(t_map, [t_map.shape[0], h, w, 3])
    # t_map_show = (t_map_show - t_map_show.min()) / (t_map_show.max() - t_map_show.min() + 1e-10)
    # if sv_path is not None:
        # imageio.mimwrite(sv_path + '/coarse_rgb.mp4', to8b(rgb_map), fps=30, quality=8)
        # imageio.mimwrite(sv_path + '/coarse_depth.mp4', to8b(t_map_show), fps=30, quality=8)

    # if args.N_samples_fine > 0:
        # rgb_map_fine, t_map_fine = np.array(rgb_map_fine).reshape([-1, h, w, 3]), np.array(t_map_fine).reshape([-1, h, w, 1])
        # t_map_show = np.broadcast_to(t_map_fine, [t_map_fine.shape[0], h, w, 3])
        # t_map_show = (t_map_show - t_map_show.min()) / (t_map_show.max() - t_map_show.min() + 1e-10)
        # if sv_path is not None:
            # imageio.mimwrite(sv_path + '/fine_rgb.mp4', to8b(rgb_map), fps=30, quality=8)
            # imageio.mimwrite(sv_path + '/fine_depth.mp4', to8b(t_map_show), fps=30, quality=8)

    # return rgb_map, t_map, rgb_map_fine, t_map_fine
    return rgb_map_fine, t_map_fine


def render_train_style(samp_func, model_forward, style_forward, concat_style_forward, latents_model_1,
                       dataset, args, device, sv_path=None, model_forward_fine=None, samp_func_fine=None, sigma_scale=0.):
    save_makedir(sv_path)
    latents_model_1.sigma_scale = sigma_scale
    # latents_model_2.rescale_sigma(sigma_scale=sigma_scale)
    frame_num, h, w = dataset.frame_num, dataset.h, dataset.w
    dataset.mode = 'train_style'

    # The largest factor of h*w closest to chunk.
    batch_size = args.chunk
    while int(h * w) % batch_size != 0:
        batch_size -= 1
    # Iteration times of each image and current iteration
    iter_img = int(h * w / batch_size)
    iter = 0
    img_count = 0
    print('Pick batch size: ', batch_size)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=args.num_workers, pin_memory=(args.num_workers > 0))

    pred_rgb, gt_rgb, pred_t = [], [], []
    pred_rgb_fine, pred_t_fine = [], []
    # Iteration
    for batch_idx, batch_data in enumerate(tqdm(dataloader)):

        style_id_check = img_count // frame_num
        image_id_check = img_count % frame_num
        img_path_check = sv_path + '/style_%05d_fine_%05d.png' % (style_id_check, image_id_check)
        if not os.path.exists(img_path_check):
            # To Device as Tensor
            for key in batch_data:
                batch_data[key] = torch.tensor(batch_data[key].numpy()).to(device)

            # Get batch data
            rgb_gt, rays_o, rays_d, rgb_origin = batch_data['rgb_gt'], batch_data['rays_o'], batch_data['rays_d'], batch_data['rgb_origin']
            style_id, frame_id = batch_data['style_id'].long().detach().cpu(), batch_data['frame_id'].long().detach().cpu()

            # Sample
            pts, ts = samp_func(rays_o=rays_o, rays_d=rays_d, N_samples=args.N_samples, near=dataset.near, far=dataset.far, perturb=True)
            ray_num, pts_num = rays_o.shape[0], args.N_samples
            rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])

            # Forward
            ret = model_forward(pts=pts, dirs=rays_d_forward)
            base_remap_features, pts_sigma, pts_embed = ret['base_remap'], ret['sigma'], ret['pts']
            # style module 1
            first_style_latents = latents_model_1(style_ids=style_id, frame_ids=frame_id, type=args.dataset_type)
            first_style_latents_forward = first_style_latents.unsqueeze(1).expand([ray_num, pts_num, first_style_latents.shape[-1]])
            first_ret_style = concat_style_forward(x=pts_embed, latent=first_style_latents_forward)
            concat_features = first_ret_style['concat_features']
            # concated_features: [1024, 64, 512]
            concated_features = torch.concat((base_remap_features, concat_features), dim=-1)
            # Stylize
            # style_latents = latents_model_2(style_ids=style_id, frame_ids=frame_id)
            style_latents = torch.mean(first_style_latents, dim=1, keepdims=True)
            style_latents_forward = torch.unsqueeze(style_latents, dim=2).expand([ray_num, pts_num, first_style_latents.shape[-1]])
            ret_style = style_forward(x=pts_embed, concated=concated_features, latent=style_latents_forward)
            pts_rgb_style = ret_style['rgb']
            # Composition
            rgb_exp_style, t_exp, weights = alpha_composition(pts_rgb_style, pts_sigma, ts, 0)

            # pred_rgb.append(torch.clamp(rgb_exp_style, 0., 1.).detach().cpu().numpy())
            # pred_t.append(t_exp.detach().cpu().numpy())
            # gt_rgb.append(rgb_gt.detach().cpu().numpy())

            if args.N_samples_fine > 0:
                # Sample
                pts_fine, ts_fine = samp_func_fine(rays_o, rays_d, ts, weights, args.N_samples_fine)
                pts_num = args.N_samples + args.N_samples_fine
                rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
                # Forward
                ret = model_forward_fine(pts=pts_fine, dirs=rays_d_forward)
                base_remap_features_fine, pts_sigma_fine, pts_embed_fine = ret['base_remap'], ret['sigma'], ret['pts']
                # style module 1
                first_style_latents_forward = first_style_latents.unsqueeze(1).expand(
                    [ray_num, pts_num, first_style_latents.shape[-1]])
                first_ret_style_fine = concat_style_forward(x=pts_embed_fine, latent=first_style_latents_forward)
                concat_features_fine = first_ret_style_fine['concat_features']
                # concated_features_fine: [1024, 128, 512]
                concated_features_fine = torch.concat((base_remap_features_fine, concat_features_fine), dim=-1)
                # Stylize
                style_latents_forward = torch.unsqueeze(style_latents, dim=2).expand([ray_num, pts_num, first_style_latents.shape[-1]])
                ret_style = style_forward(x=pts_embed_fine, concated=concated_features_fine, latent=style_latents_forward)
                pts_rgb_style_fine = ret_style['rgb']
                # Composition
                rgb_exp_style_fine, t_exp_fine, _ = alpha_composition(pts_rgb_style_fine, pts_sigma_fine, ts_fine, 0)
                pred_rgb_fine.append(torch.clamp(rgb_exp_style_fine, 0., 1.).detach().cpu().numpy())
                pred_t_fine.append(t_exp_fine.detach().cpu().numpy())

        iter += 1

        # Complete per image
        if iter == iter_img:
            if not os.path.exists(img_path_check):
                # Reshape
                # pred_rgb = np.concatenate(pred_rgb, axis=0).reshape([h, w, 3])
                # pred_t = np.concatenate(pred_t, axis=0).reshape([h, w])
                # gt_rgb = np.concatenate(gt_rgb, axis=0).reshape([h, w, 3])
                #
                # # Broadcast
                # pred_t = np.broadcast_to(pred_t[..., np.newaxis], [h, w, 3])
                # # Normalize
                # pred_t = (pred_t - np.min(pred_t)) / (np.max(pred_t) - np.min(pred_t))

                # To 255
                # pred_rgb, pred_t = np.array(pred_rgb * 255, np.int32), np.array(pred_t * 255, np.int32)
                # gt_rgb = np.array(gt_rgb * 255, np.int32)

                # Saving images
                style_id = img_count // frame_num
                image_id = img_count % frame_num
                # imageio.imwrite(sv_path + '/style_%05d_coarse_%05d.png' % (style_id, image_id), to8b(pred_rgb))
                # imageio.imwrite(sv_path + '/style_%05d_coarse_depth_%05d.png' % (style_id, image_id), to8b(pred_t))
                # imageio.imwrite(sv_path + '/style_%05d_2d_%05d.png' % (style_id, image_id), to8b(gt_rgb))

                if args.N_samples_fine > 0:
                    pred_rgb_fine = np.concatenate(pred_rgb_fine, axis=0).reshape([h, w, 3])
                    pred_t_fine = np.concatenate(pred_t_fine, axis=0).reshape([h, w])
                    pred_t_fine = np.broadcast_to(pred_t_fine[..., np.newaxis], [h, w, 3])
                    pred_t_fine = (pred_t_fine - pred_t_fine.min()) / (pred_t_fine.max() - pred_t_fine.min())
                    pred_rgb_fine, pred_t_fine = np.array(pred_rgb_fine * 255, np.int32), np.array(pred_t_fine * 255, np.int32)
                    imageio.imwrite(sv_path + '/style_%05d_fine_%05d.png' % (style_id, image_id), to8b(pred_rgb_fine))
                    imageio.imwrite(sv_path + '/style_%05d_fine_depth_%05d.png' % (style_id, image_id), to8b(pred_t_fine))
                img_count += 1
                print("Finish %d Image ..." % img_count)
                iter = 0
                # pred_rgb, gt_rgb, pred_t, gt_t, depth_masks = [], [], [], [], []
                pred_rgb_fine, pred_t_fine = [], []
            else:
                img_count += 1
                print("Skip %d Image ..." % img_count)
                iter = 0
                pred_rgb, gt_rgb, pred_t, gt_t, depth_masks = [], [], [], [], []
                pred_rgb_fine, pred_t_fine = [], []

