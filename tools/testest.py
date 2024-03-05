import torch

# 定义要平移的距离
shift_x = 1
shift_y = 0
img_bev = torch.tensor([[[[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 16]]]])

# 对img_bev进行x和y方向上的平移
shifted_img_bev = torch.roll(img_bev, shifts=(shift_x, shift_y), dims=(3, 2))
import pdb; pdb.set_trace()