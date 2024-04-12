import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from mmcv.ops.group_points import QueryAndGroup
from mmcv.ops import gather_points
from mmcv.ops.points_sampler import DFPSSampler

from mmdet3d.models.builder import BACKBONES

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


@BACKBONES.register_module()
class PointTransformer_MAE(nn.Module):
    def __init__(self,
                 in_channels,
                 trans_dim,
                 depth,
                 drop_path_rate,
                 num_heads,
                 group_k,
                 patch_num,
                 encoder_dims,
                 det_token_num,
                 det_token_init_type='random',
                 det_init_k=None,
                 bbox_mean_size=None,
                 **kwargs
                 ):
        '''Point Transformer backbone for 3D detection
            (stole the impl from PointBERT(https://github.com/lulutang0608/Point-BERT.git))

        Args:
            trans_dim (int): the dimension of features in transformer
            depth (int): the num of transformer encoder layers
            drop_path_rate (float): the rate used for DropPath
            num_heads (int): the num of heads for Multi-Head Attention
            group_k (int): the k for kNN sampling used in Patchify
            patch_num (int): the num of patches of the input point cloud
            encoder_dims (int): the tmp dimension of Patchify
            det_token_num (int): the num of [DET] tokens,
                if equals 0, no [DET] token will be used,
                else, [DET] tokens will be initialized as zeros and attached to
                the end of input tensors (the position embedding of [DET] tokens
                will be initialized randomly by using torch.randn function)
        '''
        super().__init__()

        warning_str = '*' * 30 + '\n'
        warning_str = warning_str * 5
        warning_str = warning_str + '[Warning] Using Vanilla MAE!\n'
        warning_str = warning_str + (('*') * 30 + '\n') * 5
        print(warning_str)

        self.in_channels = in_channels
        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.num_heads = num_heads
        self.det_token_num = det_token_num
        self.det_init_k = det_init_k

        self.group_size = group_k
        self.patch_num = patch_num
        # grouper
        self.group_divider = Group(num_group=self.patch_num, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = encoder_dims
        self.encoder = Encoder(in_channel=self.in_channels, encoder_channel=self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.use_random_gt = kwargs['use_random_gt']
        self.random_gt_shift = kwargs['random_gt_shift']
        self.share_mlp = kwargs['share_mlp']
        self.encoding_gt_label = kwargs['encoding_gt_label']
        self.encoding_size_prior = kwargs['encoding_size_prior']

        # img token embedding cfgs
        self.use_img_tokens = kwargs['use_img_tokens']
        self.img_channels = kwargs.get('img_channels', None)
        self.img_size = kwargs.get('img_size', None)
        self.img_patch_size = kwargs.get('img_patch_size', None)
        self.use_pretrained = kwargs.get('pretrained', False)
        self.pretrained_model_path = kwargs.get('pretrained_model', None)

        if self.use_img_tokens:
            if isinstance(self.img_patch_size, int):
                self.img_patch_size = (self.img_patch_size, self.img_patch_size)

            self.img_token_embed = PatchEmbed2D(self.img_channels, self.trans_dim, self.img_patch_size)
            
            self.img_patch_num = (self.img_size[0] // self.img_patch_size[0]) * (self.img_size[1] // self.img_patch_size[1])            
            self.img_pos_embed = nn.Parameter(torch.randn(1, self.img_patch_num, self.trans_dim))
        ######################## add [DET] tokens here ################################
        if self.det_token_num > 0:
            print(
                f'{self.__class__.__name__}: adding {self.det_token_num} [DET] tokens. Init type: {det_token_init_type}')
            self.det_token_init_type = det_token_init_type.lower()
            if self.det_token_init_type == 'random':
                self.det_token = nn.Parameter(torch.zeros(1, self.det_token_num, self.trans_dim))
                nn.init.trunc_normal_(self.det_token, std=.02)
                self.det_pos = nn.Parameter(torch.zeros(1, self.det_token_num, self.trans_dim))
                nn.init.trunc_normal_(self.det_pos, std=.02)
            elif self.det_token_init_type.startswith('gt'):
                warning_str = '*' * 30 + '\n'
                warning_str = warning_str * 5
                warning_str = warning_str + '[Warning] Using GT infos!\n'
                warning_str = warning_str + (('*') * 30 + '\n') * 5
                print(warning_str)

                self.input_dim = (4 + self.encoding_size_prior*3) if self.encoding_gt_label else (3+self.encoding_size_prior*3) #[x,y,z,label]

                if self.det_token_init_type == 'gt':
                    self.det_token_init = TokenInitializer(self.det_token_num, self.trans_dim, input_dim = self.input_dim, use_fps=False, share_mlp=self.share_mlp)
                elif self.det_token_init_type == 'gt_knn':
                    self.det_token_init = KnnGTTokenInitializer(self.det_init_k, self.trans_dim, input_dim=self.input_dim)
            elif self.det_token_init_type == 'fps':

                self.det_token_init = TokenInitializer(self.det_token_num, self.trans_dim, share_mlp=self.share_mlp)
            else:
                raise NotImplementedError(
                    f'{self.__class__.__name__}: {self.det_token_init_type} has not been implemented yet.')
        ###############################################################################

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        self.blocks = VisionTransformer(
            patch_size=16, embed_dim=self.trans_dim, depth=self.depth, num_heads=self.num_heads, mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # load pretrained if necessary
        if self.use_pretrained:
            assert self.pretrained_model_path is not None
            if self.pretrained_model_path.endswith('.npz'):  # only handle .npz weight here
                print('*' * 25 + f'load pretrain weight from {self.pretrained_model_path}' + '*' *25)
                self.blocks.load_pretrained(self.pretrained_model_path)
                print('*' * 60)
            else:
                print(f'Not .npz weight, skip loading weight in {type(self).__name__}')

        self.norm = nn.LayerNorm(self.trans_dim)

        #################### Using bbox_mean_size as input tokens ##############################
        if bbox_mean_size is not None:
            self.use_mean_size = True
            bbox_mean_size = torch.from_numpy(np.array(bbox_mean_size, dtype=np.float32))

            self.bbox_mean_size = bbox_mean_size.unsqueeze(0)  # 1 x num_class x 3

            self.mean_size_encoder = Mlp(3, self.encoder_dims, self.trans_dim)
            self.mean_size_pos = nn.Parameter(torch.zeros(1, self.bbox_mean_size.shape[1], self.trans_dim))
            nn.init.trunc_normal_(self.mean_size_pos, std=0.02)
        else:
            self.use_mean_size = False
        #########################################################################################

    def forward(self, pts, gt_bboxes_3d=None, gt_labels_3d=None, img=None):
        '''Forward function of Point Transformer backbone used to extract the features
            of input point cloud

        Args:
            pts (torch.Tensor): the input point cloud, B x N x 3

        Returns:
            features (torch.Tensor): the features tensor,
                if #[DET] > 0, return all the [DET] tokens, B x #[DET] x trans_dim
                else, return all tokens, B x (1 + #[PATCH]) x trans_dim
        '''
        # divide the point cloud into 'patches'. This is important
        neighborhood, center = self.group_divider(pts[..., :3], pts[..., 3:] if pts.shape[-1] > 3 else None)
        # encoder the input cloud blocks
        # ipdb.set_trace()

        point_tokens = self.encoder(neighborhood)  # B G N
        point_tokens = self.reduce_dim(point_tokens)

        if self.use_img_tokens:
            assert img is not None
            img_tokens = self.img_token_embed(img)
            img_pos = self.img_pos_embed.expand(point_tokens.size(0), -1, -1)
            group_input_tokens = torch.cat([point_tokens, img_tokens], dim=1)
        else:
            group_input_tokens = point_tokens

        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # prepare [DET] token
        if self.det_token_num > 0:
            det_tokens_center = None
            if self.det_token_init_type == 'random':
                det_tokens = self.det_token.expand(group_input_tokens.size(0), -1, -1)
                det_pos = self.det_pos.expand(group_input_tokens.size(0), -1, -1)
            elif self.det_token_init_type.startswith('gt'):
                assert gt_bboxes_3d is not None
                # TODO: [warning] using gt info here
                gt_centers = []
                gt_num = []
                for index in range(len(gt_bboxes_3d)):
                    bbox = gt_bboxes_3d[index]
                    gravity_center = bbox.gravity_center.clone()

                    gt_fg_mask = gt_labels_3d[index] >= 0  # remove the do not care boxes
                    gravity_center = gravity_center[gt_fg_mask]

                    if self.use_random_gt:
                        gravity_center[:, 0] = gravity_center[:, 0] +  torch.from_numpy(np.random.uniform(self.random_gt_shift[0], self.random_gt_shift[1], gravity_center.shape[0]))
                        gravity_center[:, 1] = gravity_center[:, 1] +  torch.from_numpy(np.random.uniform(self.random_gt_shift[2], self.random_gt_shift[3], gravity_center.shape[0]))
                        gravity_center[:, 2] = gravity_center[:, 2] +  torch.from_numpy(np.random.uniform(self.random_gt_shift[4], self.random_gt_shift[5], gravity_center.shape[0]))

                    if self.encoding_size_prior:
                        new_gravity_center = []
                        for idx in range(3):
                            mask = gt_labels_3d[index] >= 0
                            gt_labels = gt_labels_3d[index][mask] # remove do not box
                            #print(gt_labels.shape, gravity_center.shape)
                            gt_fg_mask = gt_labels.clone() == idx
                            selectd_cls_center = gravity_center[gt_fg_mask].clone().cuda()
                            car_size = self.bbox_mean_size[:,idx,:].cuda()
                            car_size = car_size.expand(selectd_cls_center.size(0), -1)
                            new_gravity_center.append(torch.cat([selectd_cls_center, car_size], 1))
                        gravity_center = torch.cat([new_gravity_center[0], new_gravity_center[1], new_gravity_center[2]],0)
                        #print(gravity_center.shape)
                    if self.encoding_gt_label:
                        gt_labels = gt_labels_3d[index].clone().unsqueeze(1).cuda()
                        gt_fg_mask = gt_labels >= 0  # remove the do not care boxes
                        gt_labels = (gt_labels[gt_fg_mask] + 1).unsqueeze(1) # +1 to resolve the ambiguity in zero padding

                        gravity_center = gravity_center.cuda()
                        gravity_center = torch.cat([gravity_center, gt_labels], 1)

                    pad_num = self.det_token_num - gravity_center.shape[0]
                    gt_num.append( gravity_center.shape[0])
                    gt_center = F.pad(gravity_center, (
                        0, 0,
                        0, pad_num
                    )).to(pts.device)
                    gt_centers.append(gt_center)
                det_tokens_center = torch.stack(gt_centers)
                if self.det_token_init_type == 'gt':
                    det_tokens, det_pos = self.det_token_init(det_tokens_center)
                elif self.det_token_init_type == 'gt_knn':
                    det_tokens, det_pos = self.det_token_init(pts, det_tokens_center, gt_num)
            elif self.det_token_init_type == 'fps':
                det_tokens, det_pos = self.det_token_init(center)
            else:
                raise NotImplementedError(
                    f'{self.__class__.__name__}: {self.det_token_init_type} has not been implemented yet.')
        # add pos embedding
        pos = self.pos_embed(center)
        if self.use_img_tokens:
            pos = torch.cat([pos, img_pos], dim=1)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        # prepare bbox mean size tokens
        if self.use_mean_size:
            self.bbox_mean_size = self.bbox_mean_size.to(group_input_tokens.device)
            mean_size = self.bbox_mean_size.expand(group_input_tokens.size(0), -1, -1)  # B x num_class x 3
            mean_size_pos = self.mean_size_pos.expand(group_input_tokens.size(0), -1, -1)

            mean_size_tokens = self.mean_size_encoder(mean_size)  # B x num_class x trans_dim
            x = torch.cat((x, mean_size_tokens), dim=1)
            pos = torch.cat((pos, mean_size_pos), dim=1)

        # if has [DET] tokens, append them to the tail of input tensor
        if self.det_token_num > 0:
            x = torch.cat((x, det_tokens), dim=1)  # B x (1 + #[PATCH] + #[DET]) x trans_dim
            pos = torch.cat((pos, det_pos), dim=1)

        # transformer
        x = self.blocks(x, pos)
        # x = self.norm(x)

        if self.det_token_num > 0:
            x = x[:, -self.det_token_num:, :]  # only return the [DET] tokens, B x #[DET] x trans_dim
        else:
            x = x[:, :1, :]  # only return the [CLS] token, B x 1 x trans_dim

        output_dict = dict(
            features = x,
            det_token_center = det_tokens_center
        )
        return output_dict


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SharedMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channel_last=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.channel_last = channel_last

        self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        '''forward pass of SharedMLP

        Args:
            x (torch.Tensor): input with shape: B x N X C if channel_last else B x C x N
        '''
        if self.channel_last:
            x = x.transpose(-1, -2)  # B x C x N
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.channel_last:
            x = x.transpose(-1, -2)  # B x N x C
        return x


class Group(nn.Module):
    def __init__(self, num_group, group_size, normalize_xyz=False):
        super().__init__()
        self.num_group = num_group

        self.fps = DFPSSampler()
        self.grouper = QueryAndGroup(
            max_radius=None,  # to use kNN sampling
            sample_num=group_size,
            normalize_xyz=normalize_xyz,
        )
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        center_idx = self.fps(points=xyz, npoint=self.num_group, features=None)
        center = gather_points(xyz.transpose(1, 2).contiguous(), center_idx).transpose(1, 2).contiguous()
        groups = self.grouper(xyz.contiguous(), center,
                              features.transpose(-1, -2).contiguous() if features is not None else None)  # B 3 G M
        groups = groups.permute(0, 2, 3, 1).contiguous()  # B G M 3
        return groups, center


class TokenInitializer(nn.Module):
    def __init__(self, num_tokens, out_dims, input_dim = 3, use_fps=True, share_mlp=True) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.use_fps = use_fps
        if self.use_fps:
            self.fps_sampler = DFPSSampler()

        if share_mlp:
            print('''********************************using shared MLP********************************''')
            self.token_embedding = SharedMlp(input_dim, out_dims // 2, out_dims)
            self.pos_embedding = SharedMlp(input_dim, out_dims // 2, out_dims)
        else:
            self.token_embedding = Mlp(input_dim, out_dims // 2, out_dims)
            self.pos_embedding = Mlp(input_dim, out_dims // 2, out_dims)


    def forward(self, points: torch.Tensor):
        '''forward pass for token initialization

        Args:
            pts (torch.Tensor): B x N x C
        '''
        if self.use_fps:
            center_idx = self.fps_sampler(points=points, npoint=self.num_tokens, features=None)
            center = gather_points(points.transpose(1, 2).contiguous(), center_idx).transpose(1, 2).contiguous()
        else:
            center = points
        det_token = self.token_embedding(center)
        det_pos = self.pos_embedding(center)
        return det_token, det_pos


class KnnGTTokenInitializer(nn.Module):
    def __init__(
        self,
        K,
        out_dims,
        input_dim=3,
    ):
        super().__init__()

        self.knn_grouper = QueryAndGroup(
            max_radius=None,
            sample_num=K,
            normalize_xyz=False,
        )

        self.encoder = Encoder(input_dim, out_dims)
        self.pos_embedding = SharedMlp(input_dim, out_dims // 2, out_dims)

    def forward(self, points: torch.Tensor, gt_centers: torch.Tensor, gt_num):
        '''forward pass of [DET] token initialization

        Args:
            points (torch.Tensor): B x N x 3
            gt_centers (torch.Tensor): B x #[DET] x 3
            gt_num (int): the num of real gt
        '''
        points = points[:,:,0:3].contiguous()

        #padding gt centers
        new_points = []
        # for index in range(points.shape[0]):
        #     new_points.append(torch.cat([points[index,...], gt_centers[index, :max(gt_num), :]], 0).contiguous())
        # points = torch.stack(new_points).contiguous()


        groups = self.knn_grouper(points, gt_centers)
        groups = groups.permute(0, 2, 3, 1).contiguous()  # B G K 3
        groups = groups + gt_centers.unsqueeze(2) # B G K 3 + B G 1 3

        for index in range(groups.shape[0]):
            groups[index, gt_num[index]:, :, :] *=0

        det_token = self.encoder(groups)  # B G C
        det_pos = self.pos_embedding(gt_centers)
        return det_token, det_pos



class Encoder(nn.Module):
    def __init__(self, in_channel, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.in_channel = in_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, c = point_groups.shape
        assert c == self.in_channel
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class PatchEmbed2D(nn.Module):
    def __init__(
        self,
        in_channel,
        embed_dim,
        patch_size,
        flatten=True,
    ):
        super().__init__()
        self.flatten = flatten

        self.proj = nn.Conv2d(in_channel, embed_dim, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        '''forward pass of patch embedding

        Args:
            x (torch.Tensor): with shape (B, C, H, W)
        
        Return:
            patch_feats (torch.Tensor): with shape (B, N, C)
        '''
        x = self.proj(x) 
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B x N x C
        return x

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward(self, x, pos):

        x = x + pos
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        #x = self.norm(x)

        return x
