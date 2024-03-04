import torch
from torch import nn
from pcdet.ops.bev_pool import bev_pool
import numpy as np
from scipy.spatial import cKDTree

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx
class FastDepthDeformableAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1x1 = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, query_depth, neighbors_depth):
        # query_depth[N, 1, 256, 704]    
        # neighbors_depth [N, 8, 256, 704]    
        N, S, H, W = neighbors_depth.shape
        
        # 计算深度差值 
        depth_diff = query_depth - neighbors_depth #[N, 8, 256, 704]
        depth_diff = depth_diff.view(N, S, H*W)   #[N, 8, 180224]
        neighbors_depth = neighbors_depth.view(N, S, H*W)#[N, 8, 180224]
        # print("depth_diff",depth_diff.shape)
        # Softmax作为weight
        weight = F.softmax(-torch.abs(depth_diff), dim=1)#[N, 8, 180224]
        # print("weight",weight.shape)
        
        
        # 直接使用neighbors的depth作为value
        out = torch.sum(weight * neighbors_depth, dim=1) 
        # print("out",out.shape)
        out = out.reshape(N, -1, 256, 704)


        out = self.conv1x1(out)  

        # print("out",out.shape)
    
        return out

class DeformableDepthLSSTransform(nn.Module):
    """
        This module implements LSS, which lists images into 3D and then splats onto bev features.
        This code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL #80
        self.image_size = self.model_cfg.IMAGE_SIZE
        self.feature_size = self.model_cfg.FEATURE_SIZE
        xbound = self.model_cfg.XBOUND
        ybound = self.model_cfg.YBOUND
        zbound = self.model_cfg.ZBOUND
        self.dbound = self.model_cfg.DBOUND
        downsample = self.model_cfg.DOWNSAMPLE
        self.noise = self.model_cfg.Noise
        self.K_graph=8
        self.patch_num=25
        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channel #80
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.attn = FastDepthDeformableAttention() 
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channel + 64, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel, out_channel, 3, stride=downsample, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        frustum = torch.stack((xs, ys, ds), -1)
        
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans, **kwargs):

        camera2lidar_rots = camera2lidar_rots.to(torch.float)
        camera2lidar_trans = camera2lidar_trans.to(torch.float)
        intrins = intrins.to(torch.float)
        post_rots = post_rots.to(torch.float)
        post_trans = post_trans.to(torch.float)

        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        
        # cam_to_lidar
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = extra_rots.view(B, 1, 1, 1, 1, 3, 3).repeat(1, N, 1, 1, 1, 1, 1) \
                .matmul(points.unsqueeze(-1)).squeeze(-1)
            
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def bev_pool(self, geom_feats, x): #[1, 6, 118, 32, 88, 3]  [1, 6, 118, 32, 88, 80]
        geom_feats = geom_feats.to(torch.float)
        x = x.to(torch.float)

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final
    def spatial_alignment_noise(self, ori_pose, severity):
        '''
        input: ori_pose 4*4
        output: noise_pose 4*4
        '''
        ct = [0.02, 0.04, 0.06, 0.08, 0.10][severity-1]*2
        cr = [0.002, 0.004, 0.006, 0.008, 0.10][severity-1]*2
        r_noise = torch.randn((3, 3), device=ori_pose.device)* cr
        t_noise = torch.randn((3), device=ori_pose.device) * ct
        ori_pose[..., :3, :3] += r_noise
        ori_pose[..., :3, 3]+= t_noise
        return ori_pose
    def get_cam_feats(self, img, depth,neighbors_depth):
        '''
        img:[1, 6, 256, 32, 88],
        depth:[1, 6, 1, 256, 704]
        '''

        B,  N,  C,    fH,   fW = img.shape
#       B:1 N:6 C:256 fH:32 fW:88
        depth = depth.view(B * N, *depth.shape[2:]) #(6, 1, 256, 704)
        neighbors_depth = neighbors_depth.view(B * N, *neighbors_depth.shape[2:]) #[6, 8, 256, 704]
        img = img.view(B * N, C, fH, fW) #(6,256,32,88)
        
        fused_depth = self.attn(depth, neighbors_depth)
        # print("fused_depth",fused_depth.shape) #(N, 1, 256, 704)

        fused_depth = self.dtransform(fused_depth)
        # print("fused_depth",fused_depth.shape) #(N, 1, 32, 88)
        # print("img",img.shape)
        # import pdb; pdb.set_trace()
        # depth=self.dual_input_dtransform(depth,neighbors_depth)#[6, 64, 32, 88],[6, 1, 32, 88] [6, 8, 32, 88]
        # depth = self.dtransform(depth) #[6, 64, 32, 88]
        # fused_depth torch.Size([12, 64, 256, 704])
        img = torch.cat([fused_depth, img], dim=1)#[6, 64+256, 32, 88]
        img = self.depthnet(img) #[6, D+C118+80, 32, 88]

        depth = img[:, : self.D].softmax(dim=1)#[6, 118, 32, 88]  self.D=118
        img = depth.unsqueeze(1) * img[:, self.D : (self.D + self.C)].unsqueeze(2) #[6, 80, 118, 32, 88]

        img = img.view(B, N, self.C, self.D, fH, fW)#[1, 6, 80, 118, 32, 88 ]
        img = img.permute(0, 1, 3, 4, 5, 2)#[1, 6, 118, 32, 88, 80]
        return img

    def cKDTree_neighbor(self,masked_coords):



        masked_coords_numpy=masked_coords.cpu().numpy()
        # 使用 np.unique 去除重复元素，同时返回唯一坐标及其在原数组中的索引
        unique_coords, unique_indices = np.unique(masked_coords_numpy, axis=0, return_index=True)

        # 构建 KD 树
        kdtree = cKDTree(unique_coords)

        # 查询示例：获取最近的8个邻居 +1是本身
        distances, neighbor_indices = kdtree.query(masked_coords_numpy, k=self.K_graph+1)
        neighbor_indices = unique_indices[neighbor_indices]  # 使用映射还原到原始数组中的索引
        neighbor_indices = neighbor_indices[:, :1]  # 去掉本身
        neighbor_values = masked_coords_numpy[neighbor_indices]


        # 将NumPy数组转换为PyTorch张量
        neighbor_indices_torch = torch.from_numpy(neighbor_indices)
        
        neighbor_values_torch = torch.from_numpy(neighbor_values)
  
        return neighbor_indices_torch,neighbor_values_torch

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                image_fpn (list[tensor]): image features after image neck

        Returns:
            batch_dict:
                spatial_features_img (tensor): bev features from image modality
        """

        x = batch_dict['image_fpn'] #([6, 256, 32, 88],[6, 256, 16, 44])
        x = x[0]#[6, 256, 32, 88]
        BN, C, H, W = x.size()#[6, 256, 32, 88]
        img = x.view(int(BN/6), 6, C, H, W)#[1, 6, 256, 32, 88]

        camera_intrinsics = batch_dict['camera_intrinsics'] #[1, 6, 4, 4]相机的内参
        camera2lidar = batch_dict['camera2lidar']#[1, 6, 4, 4]相机投影到lidar，外参

        img_aug_matrix = batch_dict['img_aug_matrix']#[1, 6, 4, 4]由于图像的增广矩阵导致的数据增强
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']#[1, 4, 4]点云的数据增强

        lidar2image = batch_dict['lidar2image']#[1, 6, 4, 4]lidar投影到图像，外参
        if not self.training:#test
            if self.noise:
                print("spatial_alignment_noise")
                lidar2image=self.spatial_alignment_noise(lidar2image,5)
                camera2lidar=self.spatial_alignment_noise(camera2lidar,5)
            else:
                print("clean")
            

        intrins = camera_intrinsics[..., :3, :3]#[1, 6, 3, 3]
        post_rots = img_aug_matrix[..., :3, :3]#[1, 6, 3, 3]
        post_trans = img_aug_matrix[..., :3, 3]#[1, 6, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]#[1, 6, 3, 3]
        camera2lidar_trans = camera2lidar[..., :3, 3]#[1, 6, 3]

        points = batch_dict['points']#[269283, 6]

        batch_size = BN // 6 #1
        #                     1             6         1         [256, 704]      6
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(points[0].device) #[1, 6, 1, 256, 704]
        neighbors_depth = torch.zeros(batch_size, img.shape[1], self.K_graph, *self.image_size).to(points[0].device) #[1, 6, 8, 256, 704]

        for b in range(batch_size):
            batch_mask = points[:,0] == b #269283
            cur_coords = points[batch_mask][:, 1:4]#[269283, 3]
            cur_img_aug_matrix = img_aug_matrix[b] #[6, 4, 4]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]#[4, 4]

            #造成depth预估错误的代码是因为lidar2image是有误差的[1, 6, 4, 4]
            cur_lidar2image = lidar2image[b] #[6, 4, 4]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3] #3
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )#[3, 269283]
            # lidar2image
            
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords) #[6, 3, 269283]
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)#[6, 3, 269283]
            # get 2d coords
            dist = cur_coords[:, 2, :] #[6, 269283]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)#用于将张量的元素限制在指定的范围内
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # do image aug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)#[6, 3, 269283]
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1) #[6, 3, 269283]
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]] #[6, 269283, 2]

            # filter points outside of images
            on_img = (#[6, 269283]
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(on_img.shape[0]):
                
                # cur_coords [6, 269283, 2]  #on_img[6, 269283]是bool形式
                masked_coords = cur_coords[c, on_img[c]].long()#masked_coords是被投影位置的索引.[20620, 2] 2是wh的区间分别是(0,255)(0,703)
                # print(masked_coords.shape)
                _,masked_coords_neighbors=self.cKDTree_neighbor(masked_coords)#(20620, 8, 2)
                # import pdb; pdb.set_trace()
                masked_dist = dist[c, on_img[c]] #[20620] #dist[6, 269283]是被投影位置的深度信息,最大是64，最小是0
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
                neighbors_depth[b, c, :, masked_coords_neighbors[:, :, 0].unsqueeze(-1), masked_coords_neighbors[:, :, 1].unsqueeze(-1)] = masked_dist.view(1, -1, 1, 1)
                
                # neighbors_depth[b, c,  masked_coords_neighbors[..., 0], masked_coords_neighbors[..., 1]] = masked_dist
                # 假设 b, c 是你要设置的 batch 和 channel 的索引


                #neighbors_depth#[1, 6, 8, 256, 704]
        
        extra_rots = lidar_aug_matrix[..., :3, :3] #[1, 3, 3]
        extra_trans = lidar_aug_matrix[..., :3, 3] #[1, 3]
        geom = self.get_geometry( #[1, 6, 118, 32, 88, 3]
            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        )
        
        # use points depth to assist the depth prediction in images
        # x = self.get_cam_neighbors_feats(img, neighbors_depth)
        x = self.get_cam_feats(img, depth,neighbors_depth) #[1, 6, self.D 118, 32, 88, 80] img[1, 6, 256, 32, 88]  depth[1, 6, 1, 256, 704]
        x = self.bev_pool(geom, x) #[1, 80, 360, 360]
        x = self.downsample(x)##[1, 80, 360, 360]
        # convert bev features from (b, c, x, y) to (b, c, y, x)
        x = x.permute(0, 1, 3, 2) #[1, 80, 180, 180]
        batch_dict['spatial_features_img'] = x
        return batch_dict