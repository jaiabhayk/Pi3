from .attention import FlashAttentionRope
from .block import BlockRope
from ..dinov2.layers import Mlp
import torch.nn as nn
from functools import partial
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
   
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                # attn_class=MemEffAttentionRope,
                attn_class=FlashAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, xpos=None):
        hidden = self.projects(hidden)
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=xpos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=xpos)
        out = self.linear_out(hidden)
        return out

class LinearPts3d (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3,):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(dec_embed_dim, (output_dim)*self.patch_size**2)

    def forward_org(self, decout, img_shape, grid_shape=None):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape #torch.Size([5, 1036, 1024])

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D, #torch.Size([5, 1036, 588])
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size) #torch.Size([5, 588, 28, 37])

        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W, torch.Size([5, 3, 392, 518])

        # permute + norm depth
        return feat.permute(0, 2, 3, 1) #torch.Size([5, 392, 518, 3])
    
    #import torch.nn.functional as F

    def forward(self, decout, img_shape, grid_shape):
        """
        Args:
            decout: list of token embeddings
            img_shape: (H, W) — full output resolution
            grid_shape: (h, w) — spatial grid size after token merging (e.g. 14×18)
        Returns:
            feat: (B, H, W, 3) — dense XYZ map
        """
        H, W = img_shape
        h, w = grid_shape
        tokens = decout[-1]  # (B, S, D)
        B, S, D = tokens.shape

        assert S == h * w, f"Expected {h*w} tokens but got {S}"

        # Project each token to 3D (XYZ)
        feat = self.proj(tokens)             # (B, S, 3)

        # Reshape to 2D token grid
        feat = feat.transpose(-1, -2).view(B, -1,h, w)
        feat = F.pixel_shuffle(feat, self.patch_size)
        feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=True)

        return feat.permute(0, 2, 3, 1)  # (B, H, W, 3)

    def forward_nkn(self, decout, img_shape):
        """
        Args:
            decout: list of token embeddings from decoder
            img_shape: (H, W) — full-resolution image target
        Returns:
            feat: (B, H, W, 3) — per-pixel 3D output
        """
        H, W = img_shape
        tokens = decout[-1]            # (B, S, D)
        B, S, D = tokens.shape

        # Project each token to XYZ
        feat = self.proj(tokens)       # (B, S, 3)

        # Infer spatial layout (must be square)
        patch_grid_h = int(S ** 0.5)
        patch_grid_w = patch_grid_h
        assert patch_grid_h * patch_grid_w == S, f"Token count {S} must form a square grid."

        # Reshape to (B, 3, h, w) for interpolation
        feat = feat.view(B, patch_grid_h, patch_grid_w, 3).permute(0, 3, 1, 2)  # (B, 3, h, w)

        # Bilinear upsample to full resolution
        feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)

        return feat.permute(0, 2, 3, 1)  # (B, H, W, 3)

    
    
    def forward_new(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        feat = self.proj(tokens)  # B, S, 588 (or maybe not...)

        try:
            feat1 = feat.transpose(-1, -2).reshape(B, -1, H // self.patch_size, W // self.patch_size)
            feat = F.pixel_shuffle(feat1, self.patch_size)
        except RuntimeError as e:
            print("⚠️ Pixel shuffle failed:", e)
            print("Tokens:", tokens.shape)
            print("Channels:", feat.shape[1])
            # fallback: mean + drop to 3 channels
            feat = feat.mean(dim=1, keepdim=True)              # (B, 1, D)
            feat = feat.transpose(1, 2).reshape(B, D, 1, 1)     # (B, D, 1, 1)
            feat = feat[:, :3]                                  # cut to (B, 3, 1, 1)
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        return feat.permute(0, 2, 3, 1)


        try:
            feat1 = feat.transpose(-1, -2).reshape(B, -1, H//self.patch_size, W//self.patch_size)
            feat = F.pixel_shuffle(feat1, self.patch_size)
        except RuntimeError as e:
            print("⚠️ Pixel shuffle failed:", e)
            print("Tokens:", tokens.shape)
            print("Channels:", feat.shape[1])
            # fallback: mean + interpolate to (H, W)
            feat = feat.mean(dim=1, keepdim=True)  # (B, 1, D)
            feat = feat.transpose(1, 2).reshape(B, feat.shape[-1], 1, 1)
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        return feat.permute(0, 2, 3, 1)  # (B, H, W, 3)
