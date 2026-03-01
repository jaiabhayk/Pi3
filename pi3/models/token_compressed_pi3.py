# pi3/models/token_compressed_pi3.py
from pi3.models.pi3 import Pi3
from pi3.models.token_merging import merge_tokens, merge_tokens_structured  # your ToMe-style merge
import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin


class Pi3WithToMe(Pi3):


    def forward(self, imgs, apply_merging=False):
        imgs = (imgs - self.image_mean) / self.image_std
        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)  # (B*N, hw, D)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        num_keep = None
        downsample_ratio = 2
        apply_merging = True
        
        if apply_merging:
            hidden, (h_new, w_new) = merge_tokens_structured(
                hidden, H, W, patch_size=14, downsample_ratio=downsample_ratio, method='mean'
            )


        # if apply_merging:
        #     #merge_ratio = getattr(self, "token_merge_keep_ratio", 1.0)  # or pass from config
        #     #num_keep = int(hidden.shape[1] * merge_ratio)
        #     #num_keep = hidden.shape[1] // 2
        #     def get_closest_square_token_count(S):
        #         print('S=', S)
        #         side = int(S ** 0.5)
        #         return side #* side  # e.g. 22×22 = 484
            
        #     print("hidden.shape=", hidden.shape)
        #     num_keep = 784 #get_closest_square_token_count(hidden.shape[1])
        #     print('num_keep=', num_keep)
        #     hidden = merge_tokens(hidden, num_keep=num_keep)  # 🟢 merge happens here
        #     #hidden = merge_tokens_structured(hidden, H, W, patch_size=14)

        # # expected_tokens = (H // self.patch_size) * (W // self.patch_size)
        # # assert hidden.shape[1] == expected_tokens, \
        # #     f"Expected {expected_tokens} tokens but got {hidden.shape[1]}"

        #hidden, pos = self.decode(hidden, N, H, W, num_keep==num_keep)

        if apply_merging:
            hidden, pos = self.decode(hidden, N, H, W, num_keep=(h_new, w_new))
        else:
            hidden, pos = self.decode(hidden, N, H, W)


        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W), grid_shape=(14, 18) if apply_merging else (H//self.patch_size,W//self.patch_size)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W), grid_shape=(14, 18) if apply_merging else (H//self.patch_size,W//self.patch_size)).reshape(B, N, H, W, -1)

            camera_hidden = camera_hidden.float()
            if apply_merging:
                camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], 14, 18).reshape(B, N, 4, 4)
            else:
                camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )
