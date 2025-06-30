import torch
import argparse
from collections import OrderedDict


def convert_and_split_state_dict(original_state_dict):
    """
    将一个包含 coarse_mlp 和 fine_mlp 的 state_dict 拆分并重命名。
    """
    coarse_net_dict = OrderedDict()
    fine_net_dict = OrderedDict()

    print("Starting conversion...")

    # 遍历原始 state_dict 的每一个参数
    for key, value in original_state_dict.items():
        new_key = None
        target_dict = None

        # 判断当前参数属于粗网络还是精细网络
        if key.startswith('model.coarse_mlp.'):
            target_dict = coarse_net_dict
            # 移除前缀 'model.coarse_mlp.'
            temp_key = key.replace('model.coarse_mlp.', '')

        elif key.startswith('model.fine_mlp.'):
            target_dict = fine_net_dict
            # 移除前缀 'model.fine_mlp.'
            temp_key = key.replace('model.fine_mlp.', '')

        else:
            # 如果有其他不相关的参数，可以选择跳过
            continue

        # --- 在这里定义重命名规则 ---
        # 规则是链式替换，顺序很重要
        temp_key = temp_key.replace('pts_linears.', 'base_layers.')
        temp_key = temp_key.replace('density_layer.', 'sigma_layer.')
        temp_key = temp_key.replace('bottleneck_layer.', 'base_remap_layer.')
        # 假设 views_linear.0 -> rgb_layers.0, rgb_layer -> rgb_layers.1
        temp_key = temp_key.replace('views_linear.0.', 'rgb_layers.0.')
        temp_key = temp_key.replace('rgb_layer.', 'rgb_layers.1.')

        # 添加新的 'net.' 前缀
        new_key = 'net.' + temp_key
        # ---------------------------

        # 将重命名后的键和原始值存入对应的字典
        if new_key and target_dict is not None:
            target_dict[new_key] = value
            print(f"Converted: '{key}'  ->  '{new_key}'")

    return coarse_net_dict, fine_net_dict


def main():
    parser = argparse.ArgumentParser(description="Split and convert NeRF checkpoint.")
    parser.add_argument('input_path', type=str, help="Path to the source checkpoint file.")
    parser.add_argument('output_path', type=str, help="Path to save the new checkpoint file.")
    args = parser.parse_args()

    # 加载原始 checkpoint
    print(f"Loading source checkpoint from: {args.input_path}")
    source_checkpoint = torch.load(args.input_path, map_location='cpu')

    # 提取原始 state_dict
    # 假设它在 'state_dict' 键下，如果不是，请根据实际情况修改
    if 'state_dict' not in source_checkpoint:
        raise KeyError("Could not find 'state_dict' key in the checkpoint.")
    original_state_dict = source_checkpoint['state_dict']

    # 执行转换
    coarse_state_dict, fine_state_dict = convert_and_split_state_dict(original_state_dict)
    # 创建一个新的 checkpoint 来保存两个 state_dict
    new_checkpoint = {
        'model': coarse_state_dict,
        'model_fine': fine_state_dict,
        # 你也可以保留原始checkpoint中的其他信息，比如 epoch
        # 'epoch': source_checkpoint.get('epoch', -1),
        'global_step': 120001,
    }

    # 保存新的 checkpoint
    print(f"\nSaving converted checkpoint to: {args.output_path}")
    torch.save(new_checkpoint, args.output_path)

    print("\nConversion successful!")
    print(f"Coarse network has {len(coarse_state_dict)} parameters.")
    print(f"Fine network has {len(fine_state_dict)} parameters.")


if __name__ == '__main__':
    main()