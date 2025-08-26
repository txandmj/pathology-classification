
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class DiagnosticPathologyDataset(Dataset):
    """病理图像数据集类"""
    """
        病理图像数据集类 - 详细解释版本

        核心设计理念：
        1. 将少量大图像转换为大量小patch
        2. 处理不同比例尺的图像
        3. 动态生成训练样本
    """
    def __init__(self, image_paths, labels, transform=None, patch_size=224,
                 patches_per_image=3, seed=42):
        """
        初始化数据集
        参数解释：
        - image_paths: 图像文件路径列表 ['img1.jpg', 'img2.jpg', ...]
        - labels: 对应的标签列表 [0, 1, 0, 1, ...]  (0=non-LNM, 1=LNM)
        - transform: 数据增强变换
        - patch_size: 每个patch的尺寸 (默认512x512)
        - patches_per_image: 每张原图提取多少个patch
        - scale_normalizer: 比例尺归一化器
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.seed = seed

        # 确保每个图像的patch提取是确定性的
        np.random.seed(seed)

        # 生成patch索引 - 确保没有重叠
        self.patch_indices = []
        for img_idx in range(len(image_paths)):
            for patch_idx in range(patches_per_image):
                self.patch_indices.append((img_idx, patch_idx))

        print(f"Created dataset: {len(image_paths)} images → {len(self.patch_indices)} patches")

    def __len__(self):
        """
        数据集大小 = 原图数量 × 每张图的patch数
        30张原图 × 3个patch = 90个训练样本
        """
        return len(self.patch_indices)

    def extract_deterministic_patches(self, image, num_patches, img_idx):
        """确定性patch提取 - 避免随机性导致的数据泄漏"""
        h, w = image.shape[:2]
        patches = []

        # 使用图像索引作为随机种子，确保每次提取相同的patch
        np.random.seed(self.seed + img_idx)

        # 确保图像足够大
        if h < self.patch_size or w < self.patch_size:
            image = cv2.resize(image, (max(self.patch_size * 2, w), max(self.patch_size * 2, h)))
            h, w = image.shape[:2]

        # 固定的网格位置 + 少量确定性偏移
        positions = []
        grid_size = int(np.ceil(np.sqrt(num_patches)))

        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= num_patches:
                    break

                # 基础网格位置
                base_y = (h - self.patch_size) * i // max(1, grid_size - 1)
                base_x = (w - self.patch_size) * j // max(1, grid_size - 1)

                # 小幅度确定性偏移（基于图像索引）
                offset_y = ((img_idx * 13) % 50) - 25  # -25到+25的偏移
                offset_x = ((img_idx * 17) % 50) - 25

                y = np.clip(base_y + offset_y, 0, h - self.patch_size)
                x = np.clip(base_x + offset_x, 0, w - self.patch_size)

                positions.append((y, x))

        # 提取patches
        for y, x in positions:
            patch = image[y:y + self.patch_size, x:x + self.patch_size]
            if patch.shape[0] == self.patch_size and patch.shape[1] == self.patch_size:
                patches.append(patch)

        return patches[:num_patches]

    def __getitem__(self, idx):
        img_idx, patch_idx = self.patch_indices[idx]
        image_path = self.image_paths[img_idx]
        label = self.labels[img_idx]

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            # 如果读取失败，创建标识性的dummy图像
            image = np.full((self.patch_size * 2, self.patch_size * 2, 3),
                            (img_idx * 50) % 255, dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 确定性patch提取
        patches = self.extract_deterministic_patches(image, self.patches_per_image, img_idx)

        if len(patches) == 0:
            patch = cv2.resize(image, (self.patch_size, self.patch_size))
        else:
            patch = patches[patch_idx % len(patches)]

        # 转换为PIL
        patch = Image.fromarray(patch)

        if self.transform:
            patch = self.transform(patch)

        return patch, label, img_idx  # 返回原始图像索引用于调试

def get_transforms():
    """获取数据增强变换"""

    # 训练时的强数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证/测试时的变换
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform