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


@BACKBONES.register_module()
class PointTransformer(nn.Module):
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

        self.in_channels = in_channels
        self.trans_dim = trans_dim
        self.depth = depth 
        self.drop_path_rate = drop_path_rate 
        self.num_heads = num_heads 
        self.det_token_num = det_token_num

        self.group_size = group_k
        self.patch_num = patch_num
        # grouper
        self.group_divider = Group(num_group = self.patch_num, group_size = self.group_size)
        # define the encoder
        self.encoder_dims = encoder_dims
        self.encoder = Encoder(in_channel=self.in_channels, encoder_channel=self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        ######################## add [DET] tokens here ################################
        if self.det_token_num > 0:
            print(f'{self.__class__.__name__}: adding {self.det_token_num} [DET] tokens. Init type: {det_token_init_type}')
            self.det_token_init_type = det_token_init_type.lower()
            if self.det_token_init_type == 'random':
                self.det_token = nn.Parameter(torch.zeros(1, self.det_token_num, self.trans_dim))
                nn.init.trunc_normal_(self.det_token, std=.02)
                self.det_pos = nn.Parameter(torch.zeros(1, self.det_token_num, self.trans_dim))
                nn.init.trunc_normal_(self.det_pos, std=.02)
            elif self.det_token_init_type == 'gt':
                warning_str = '*' * 30 + '\n'
                warning_str = warning_str * 5
                warning_str = warning_str + '[Warning] Using GT infos!\n'
                warning_str = warning_str + (('*') * 30 + '\n') * 5
                print(warning_str)
                self.det_token_init = TokenInitializer(self.det_token_num, self.trans_dim, use_fps=False)
            elif self.det_token_init_type == 'fps':
                self.det_token_init = TokenInitializer(self.det_token_num, self.trans_dim)
            else:
                raise NotImplementedError(f'{self.__class__.__name__}: {self.det_token_init_type} has not been implemented yet.')
        ###############################################################################

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

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

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        # ipdb.set_trace()
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head') \
                and not k.startswith('transformer_q.encoder'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]


        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print('missing_keys', 'Transformer', incompatible.missing_keys)
        #     print(
        #         get_missing_parameters_message(incompatible.missing_keys),
        #         'Transformer'
        #     )
        if incompatible.unexpected_keys:
            print('unexpected_keys', 'Transformer', incompatible.unexpected_keys)
        #     print(
        #         get_unexpected_parameters_message(incompatible.unexpected_keys),
        #         'Transformer'
        #     )

        print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')


    def forward(self, pts, gt_bboxes_3d=None):
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

        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # prepare [DET] token
        if self.det_token_num > 0:
            if self.det_token_init_type == 'random':
                det_tokens = self.det_token.expand(group_input_tokens.size(0), -1, -1)
                det_pos = self.det_pos.expand(group_input_tokens.size(0), -1, -1)
            elif self.det_token_init_type == 'gt':
                assert gt_bboxes_3d is not None
                # TODO: [warning] using gt info here
                gt_centers = []
                for bbox in gt_bboxes_3d:
                    pad_num = self.det_token_num - bbox.tensor.shape[0]
                    gt_center = F.pad(bbox.gravity_center, (
                        0, 0,
                        0, pad_num
                    )).to(pts.device)
                    gt_centers.append(gt_center)
                gt_bboxes_center = torch.stack(gt_centers)  # B x N x C
                det_tokens, det_pos = self.det_token_init(gt_bboxes_center)
            elif self.det_token_init_type == 'fps':
                det_tokens, det_pos = self.det_token_init(center)
            else:
                raise NotImplementedError(f'{self.__class__.__name__}: {self.det_token_init_type} has not been implemented yet.')
        # add pos embedding
        pos = self.pos_embed(center)
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
        x = self.norm(x)

        if self.det_token_num > 0:
            x = x[:, -self.det_token_num:, :]  # only return the [DET] tokens, B x #[DET] x trans_dim
        else:
            x = x[:, :1, :]  # only return the [CLS] token, B x 1 x trans_dim
        return x  

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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
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
    
    def forward(self, xyz: torch.Tensor, features: torch.Tensor=None):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        center_idx = self.fps(points=xyz, npoint=self.num_group, features=None)
        center = gather_points(xyz.transpose(1, 2).contiguous(), center_idx).transpose(1, 2).contiguous()
        groups = self.grouper(xyz.contiguous(), center, features.transpose(-1, -2).contiguous() if features is not None else None)  # B 3 G M
        groups = groups.permute(0, 2, 3, 1).contiguous()  # B G M 3
        return groups, center


class TokenInitializer(nn.Module):
    def __init__(self, num_tokens, out_dims, use_fps=True) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.use_fps = use_fps
        if self.use_fps:
            self.fps_sampler = DFPSSampler()
        # self.token_embedding = Mlp(3, out_dims // 2, out_dims)
        # self.pos_embedding = Mlp(3, out_dims // 2, out_dims)
        self.token_embedding = SharedMlp(3, out_dims // 2, out_dims)
        self.pos_embedding = SharedMlp(3, out_dims // 2, out_dims)

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
        bs, g, n , c = point_groups.shape
        assert c == self.in_channel
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Decoder(nn.Module):
    def __init__(self, encoder_channel, num_fine):
        super().__init__()
        self.num_fine = num_fine
        self.grid_size = 2
        self.num_coarse = self.num_fine // 4
        assert num_fine % 4 == 0

        self.mlp = nn.Sequential(
            nn.Linear(encoder_channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(encoder_channel + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2) # 1 2 S


    def forward(self, feature_global):
        '''
            feature_global : B G C
            -------
            coarse : B G M 3
            fine : B G N 3
        
        '''
        bs, g, c = feature_global.shape
        feature_global = feature_global.reshape(bs * g, c)

        coarse = self.mlp(feature_global).reshape(bs * g, self.num_coarse, 3) # BG M 3

        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1) # BG (M) S 3
        point_feat = point_feat.reshape(bs * g, self.num_fine, 3).transpose(2, 1) # BG 3 N

        seed = self.folding_seed.unsqueeze(2).expand(bs * g, -1, self.num_coarse, -1) # BG 2 M (S)
        seed = seed.reshape(bs * g, -1, self.num_fine).to(feature_global.device)  # BG 2 N

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_fine) # BG 1024 N
        feat = torch.cat([feature_global, seed, point_feat], dim=1) # BG C N
    
        center = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1) # BG (M) S 3
        center = center.reshape(bs * g, self.num_fine, 3).transpose(2, 1) # BG 3 N

        fine = self.final_conv(feat) + center   # BG 3 N
        fine = fine.reshape(bs, g, 3, self.num_fine).transpose(-1, -2)
        coarse = coarse.reshape(bs, g, self.num_coarse, 3)
        return coarse, fine


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


if __name__ == '__main__':
    B, N, C = 2, 2048, 3
    points = torch.randn([B, N, C]).cuda()

    transformer = PointTransformer(
        trans_dim=384,
        depth=12,
        drop_path_rate=0.1,
        num_heads=6,
        group_k=32,
        patch_num=128,
        encoder_dims=256,
        det_token_num=100
    ).cuda()

    out = transformer(points)
    print(out.shape)