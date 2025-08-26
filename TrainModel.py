import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def get_conservative_transforms():
    """更保守的数据变换"""

    # 训练时适度的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),  # 使用中心裁剪而非随机裁剪
        transforms.RandomHorizontalFlip(p=0.3),  # 减少随机性
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 减少颜色变化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 验证时完全确定性
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_conservative_model(model, train_loader, val_loader, num_epochs=20, device='cuda'):
    """保守的训练策略"""

    model = model.to(device)

    # 非常保守的优化设置
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-3)  # 极低学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    # 标准交叉熵 + 标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'train_acc': [], 'val_acc': []}

    best_auc = 0
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_targets = []

        for batch_idx, (data, target, img_indices) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # 强梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            # 收集预测用于AUC计算
            pred_probs = torch.softmax(output, dim=1)[:, 1].cpu().detach().numpy()
            train_preds.extend(pred_probs)
            train_targets.extend(target.cpu().numpy())

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []
        val_img_indices = []

        with torch.no_grad():
            for data, target, img_indices in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

                pred_probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                val_preds.extend(pred_probs)
                val_targets.extend(target.cpu().numpy())
                val_img_indices.extend(img_indices.cpu().numpy())

                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        # 计算指标
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        # 计算AUC（确保有两个类别）
        if len(set(train_targets)) > 1:
            train_auc = roc_auc_score(train_targets, train_preds)
        else:
            train_auc = 0.5

        if len(set(val_targets)) > 1:
            val_auc = roc_auc_score(val_targets, val_preds)
        else:
            val_auc = 0.5

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train AUC: {train_auc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}')
        print(f'  Overfitting Gap: {val_loss - train_loss:.4f}')

        # 检查过拟合警告
        if val_auc == 1.0:
            print(f'  ⚠️  WARNING: Perfect AUC detected! Possible overfitting.')

        scheduler.step()

        # 早停（但不保存完美模型）
        if val_auc > best_auc and val_auc < 0.98:  # 拒绝接近完美的模型
            best_auc = val_auc
            torch.save(model.state_dict(), f'conservative_model.pth')
            patience_counter = 0
            print(f'  ✓ New best model saved (AUC: {best_auc:.4f})')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        print('-' * 60)

    return model, history
