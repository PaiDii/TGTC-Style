import VGGNet
from rendering import *

def save_rgb_array_as_image(rgb_data, target_shape, filename):
    """
    将输入的 RGB 数据（假定原始对应 512 个元素）重塑、调整范围并保存为图像文件。

    Args:
        rgb_data: 输入的 NumPy 数组或可转换为数组的对象，假定原始形状类似 (512, 3)。
        target_shape: 目标图像形状，例如 (16, 32, 3)。
        filename: 保存图像的文件名 (例如 'output.png')。
    """
    try:
        # 1. 确保是 NumPy 数组
        rgb_array = np.asarray(rgb_data)
        print(f"\nProcessing '{filename}': Original data shape: {rgb_array.shape}")

        # 2. 检查原始数据是否能 reshape 成目标形状
        expected_size = np.prod(target_shape)
        if rgb_array.size != expected_size:
            raise ValueError(
                f"Input data size ({rgb_array.size}) does not match target size ({expected_size}) for shape {target_shape}")

        # 3. Reshape 数据
        image_data = rgb_array.reshape(target_shape)
        print(f"Reshaped data to shape: {image_data.shape}")

        # 4. 检查数据类型和范围，转换为 uint8 [0, 255]
        min_val, max_val = np.min(image_data), np.max(image_data)
        dtype = image_data.dtype
        print(f"Data range before scaling: min={min_val:.4f}, max={max_val:.4f}, dtype={dtype}")

        image_data_uint8 = None
        if dtype == np.uint8:
            print("Data is already uint8. Assuming range [0, 255]. Clipping just in case.")
            image_data_uint8 = np.clip(image_data, 0, 255)  # 确保在范围内
        elif max_val <= 1.0 and min_val >= 0.0:
            print("Scaling data from [0, 1] range to [0, 255] and converting to uint8.")
            image_data_uint8 = (image_data * 255).astype(np.uint8)
        elif max_val <= 255.0 and min_val >= 0.0:
            print("Data seems to be in [0, 255] range but not uint8. Clipping and converting.")
            image_data_uint8 = np.clip(image_data, 0, 255).astype(np.uint8)
        else:
            # 其他情况，进行标准化后缩放 (这可能不是标准的RGB，但可以可视化)
            print(
                f"Warning: Data range [{min_val:.2f}, {max_val:.2f}] not standard. Normalizing and scaling to [0, 255].")
            if max_val == min_val:  # 避免除零
                normalized = np.zeros_like(image_data)
            else:
                normalized = (image_data - min_val) / (max_val - min_val)
            image_data_uint8 = (normalized * 255).astype(np.uint8)

        # 5. 使用 OpenCV 保存图像
        # cv2.imwrite 会根据文件扩展名保存。对于 PNG/JPG 等，它期望 BGR 顺序。
        # 但如果直接给它一个 (H, W, 3) 的 NumPy 数组，它通常能正确处理。
        # 为保险起见，如果确定输入是 RGB，可以转换一下。
        if image_data_uint8.shape[2] != 3:
            raise ValueError(f"Expected 3 channels after reshape, but got {image_data_uint8.shape[2]}")

        # 假设输入的 rgb_data 代表 RGB 顺序，转换为 BGR 给 imwrite
        image_to_save_bgr = cv2.cvtColor(image_data_uint8, cv2.COLOR_RGB2BGR)

        success = cv2.imwrite(filename, image_to_save_bgr)

        if success:
            print(f"Image saved successfully to: {filename}")
        else:
            print(f"Error: OpenCV (cv2.imwrite) failed to save the image '{filename}'. Check path/permissions.")

    except ImportError:
        print("Error: OpenCV (cv2) library not found. Please install it: pip install opencv-python")
    except ValueError as e:
        print(f"Error processing data for '{filename}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving '{filename}': {e}")

rgb_exp_style_fine2 = cv2.imread('./b2.png')
rgb_origin2 = cv2.imread('./b1.png')
similarity_result = VGGNet.cosine_similarity(torch.from_numpy(rgb_exp_style_fine2).reshape(-1, 3).float(), torch.from_numpy(rgb_origin2).reshape(-1, 3).float())
print("Cosine similarity calculated.")

# 2. 数据处理与转换 (确保是 numpy array)
if hasattr(similarity_result, 'detach'):
    similarity_vector = similarity_result.detach().cpu().numpy()
else:
    similarity_vector = np.asarray(similarity_result)

# 检查形状
print(f"Original similarity vector shape: {similarity_vector.shape}")
print(
    f"Data range: min={np.min(similarity_vector):.4f}, max={np.max(similarity_vector):.4f}")

# 3. 重塑为 (16, 32)
target_shape = (756, 1008)
similarity_heatmap_data = similarity_vector.reshape(target_shape)
vmin = np.min(similarity_heatmap_data)  # 或使用固定值如 0.0
vmax = np.max(similarity_heatmap_data)  # 或使用固定值如 1.0
if vmax == vmin:  # 处理数据全相等的情况
    normalized_data_01 = np.zeros_like(similarity_heatmap_data)
else:
    normalized_data_01 = (similarity_heatmap_data - vmin) / (vmax - vmin)

# 2. 缩放到 [0, 255] 并转换为 8位无符号整数 (uint8)
#    这是 OpenCV applyColorMap 函数要求的输入格式
normalized_data_uint8 = (normalized_data_01 * 255).astype(np.uint8)

# 3. 应用 OpenCV 的内建颜色映射 (Colormap)
#    选择一个颜色映射方案，例如:
#    cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_JET, cv2.COLORMAP_HOT,
#    cv2.COLORMAP_COOL, cv2.COLORMAP_PLASMA, cv2.COLORMAP_INFERNO,
#    cv2.COLORMAP_MAGMA, cv2.COLORMAP_CIVIDIS, cv2.COLORMAP_RAINBOW
heatmap_image_bgr = cv2.applyColorMap(normalized_data_uint8, cv2.COLORMAP_VIRIDIS)
# 注意：OpenCV 生成的彩色图像默认是 BGR 顺序，而非 RGB

# 4. 保存图像文件
#    cv2.imwrite 可以直接保存 BGR 格式的图像
output_filename_cv2 = 'heatmap_opencv_only.png'
target_image_shape = (756, 1008, 3)
success = cv2.imwrite(output_filename_cv2, heatmap_image_bgr)