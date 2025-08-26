import os
import re

class ScaleNormalizer:
    """处理不同比例尺的图像归一化"""

    def __init__(self, target_resolution_um_per_pixel=0.5):
        self.target_resolution = target_resolution_um_per_pixel
        # 使用普通函数替代lambda，避免pickle序列化问题
        self.scale_patterns = {
            r'(\d+\.?\d*)\s*mm': self._mm_to_um,  # mm to μm
            r'(\d+\.?\d*)\s*μm': self._um_to_um,  # μm
            r'(\d+\.?\d*)\s*um': self._um_to_um,  # um
            r'(\d+\.?\d*)\s*micron': self._um_to_um,  # micron
        }

    def _mm_to_um(self, x):
        """毫米转微米"""
        return float(x) * 1000

    def _um_to_um(self, x):
        """微米转微米（无变换）"""
        return float(x)

    def extract_scale_info(self, image_path):
        """从图像文件名或metadata中提取比例尺信息"""
        filename = os.path.basename(image_path)

        for pattern, converter_func in self.scale_patterns.items():
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                scale_value = converter_func(match.group(1))
                return scale_value

        # 如果无法提取比例尺，返回默认值
        return None

    def calculate_resize_factor(self, current_scale_um, image_width):
        """计算缩放因子以归一化到目标分辨率"""
        if current_scale_um is None:
            return 1.0

        current_resolution = current_scale_um / image_width
        resize_factor = current_resolution / self.target_resolution
        return resize_factor

