import torch.nn as nn
import torchvision.models as models
import warnings
warnings.filterwarnings('ignore')


class ConservativeModel(nn.Module):
    """更保守的模型 - 防止过拟合"""

    def __init__(self, num_classes=2, dropout_rate=0.6):
        super(ConservativeModel, self).__init__()

        # 使用最简单的ResNet18，并冻结部分层
        self.backbone = models.resnet18(pretrained=True)

        # 冻结前几层
        for param in list(self.backbone.parameters())[:-20]:  # 只训练最后20个参数
            param.requires_grad = False

        # 获取特征维度
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 极简分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 64),  # 进一步减少参数
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(64, num_classes)
        )

        print(f"Model created with {sum(p.numel() for p in self.parameters() if p.requires_grad)} trainable parameters")

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


