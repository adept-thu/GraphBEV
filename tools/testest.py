import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossDeformableAttention(nn.Module):
    def __init__(self):
        super(CrossDeformableAttention, self).__init__()

        # 偏移生成网络
        self.offset_conv = nn.Conv2d(336, 2, kernel_size=3, stride=1, padding=1)

        # 形变卷积
        self.deform_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, img_bev, lidar_bev):
        # 生成偏移
        offset = self.offset_conv(torch.cat([img_bev, lidar_bev], dim=1))
        print(offset.shape)

        # 将 offset 的维度调整为 [1, 180, 180, 2]
        offset = offset.permute(0, 2, 3, 1)

        # 生成形变卷积的权重
        deform_weight = F.grid_sample(lidar_bev, offset)
        print(deform_weight.shape)

        # 应用形变卷积
        deformed_feature = self.deform_conv(lidar_bev * deform_weight)

        return deformed_feature

# 创建模型实例
CrossDeformabl = CrossDeformableAttention()

# 随机生成输入张量
img_bev = torch.randn((1, 80, 180, 180))
lidar_bev = torch.randn((1, 256, 180, 180))

# 模型前向传播
output_tensor = CrossDeformabl(img_bev, lidar_bev)

# 打印输出张量的形状
print(f"Output shape: {output_tensor.shape}")
