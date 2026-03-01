import torch
import torch.nn.functional as F

def merge_tokens(features, num_keep):
    B, N, D = features.shape
    if N <= num_keep:
        return features
    normed = F.normalize(features, dim=-1)
    sim = torch.matmul(normed, normed.transpose(1, 2))  # (B, N, N)
    saliency = sim.mean(dim=-1)  # (B, N)
    keep_indices = saliency.topk(num_keep, dim=-1).indices
    return torch.stack([features[b][keep_indices[b]] for b in range(B)])

def merge_tokens_structured_old_badMayBe(tokens, H, W, patch_size=14, method="mean"):
    """
    Grid-aware merging: outputs exactly H//p × W//p tokens per image.

    Args:
        tokens: (B*N, hw, D)
        H, W: original image size
        patch_size: assumed patch size (14)
        method: 'mean' or 'center'

    Returns:
        merged: (B*N, hw_reduced, D)
    """
    B_N, hw, D = tokens.shape
    h = H // patch_size
    w = W // patch_size

    assert hw == h * w, f"Token count mismatch: got {hw}, expected {h*w}"

    tokens = tokens.reshape(B_N, h, w, D)

    if method == "mean":
        merged = tokens  # nothing to do
    elif method == "center":
        # pick center pixel from each patch
        cx = h // 2
        cy = w // 2
        merged = tokens[:, cx:cx+1, cy:cy+1, :]  # will be 1x1
    else:
        raise NotImplementedError(f"Merging method {method} not supported.")

    return merged.reshape(B_N, h * w, D)

def merge_tokens_structured(tokens, H, W, patch_size=14, method="mean", downsample_ratio=2):
    """
    Grid-aware structured token merging by spatial downsampling.

    Args:
        tokens: (B*N, hw, D)
        H, W: original image size
        patch_size: size of each patch (e.g. 14)
        method: merging method within blocks ('mean', 'center')
        downsample_ratio: spatial reduction factor (e.g. 2 means 2x2 → 1)

    Returns:
        merged: (B*N, new_hw, D)
        new_grid_shape: (h', w') token grid shape after merging
    """
    B_N, hw, D = tokens.shape
    h = H // patch_size
    w = W // patch_size
    assert hw == h * w, f"Token count mismatch: got {hw}, expected {h*w}"

    tokens = tokens.view(B_N, h, w, D)  # (B*N, h, w, D)

    # Adjust h, w to be divisible by downsample_ratio
    h_trim = h - (h % downsample_ratio)
    w_trim = w - (w % downsample_ratio)
    tokens = tokens[:, :h_trim, :w_trim, :]  # crop if needed

    h_new = h_trim // downsample_ratio
    w_new = w_trim // downsample_ratio

    # Reshape into block structure
    x = tokens.view(B_N, h_new, downsample_ratio, w_new, downsample_ratio, D)

    if method == "mean":
        merged = x.mean(dim=(2, 4))  # average over each 2x2 block
    elif method == "center":
        merged = x[:, :, downsample_ratio//2, :, downsample_ratio//2, :]
    else:
        raise NotImplementedError(f"Merging method '{method}' not supported.")

    return merged.view(B_N, h_new * w_new, D), (h_new, w_new)
