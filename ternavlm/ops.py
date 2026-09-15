import torch


def pixel_shuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """(B, N, C) with N = H*W square grid -> (B, N/scale^2, C*scale^2). Merges scale x scale patch blocks."""
    b, n, c = x.shape
    h = w = int(n ** 0.5)
    assert h * w == n, f"non-square patch grid: {n}"
    assert h % scale == 0, f"grid {h} not divisible by {scale}"
    x = x.view(b, h // scale, scale, w // scale, scale, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, (h // scale) * (w // scale), c * scale * scale)
