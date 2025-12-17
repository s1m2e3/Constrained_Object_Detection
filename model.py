import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ----------------------- utilities -----------------------

def circular_pad(x, pad: int):
    return F.pad(x, (pad, pad, pad, pad), mode='circular')

def gaussian_kernel_2d_from_Sigma(size: int, Sigma: torch.Tensor):
    """
    Discretize a normalized Gaussian from 2x2 SPD Sigma over offsets (Δx,Δy).
    Returns (size, size) kernel summing to 1.
    """
    assert size % 2 == 1
    R = size // 2
    device, dtype = Sigma.device, Sigma.dtype
    ys, xs = torch.meshgrid(
        torch.arange(-R, R+1, device=device, dtype=dtype),
        torch.arange(-R, R+1, device=device, dtype=dtype),
        indexing='ij'
    )
    X = torch.stack([xs, ys], dim=-1).reshape(-1, 2)   # (S,2), S=size^2
    Sinv = torch.linalg.inv(Sigma)                     # (2,2)
    q = torch.einsum('sd,df,sf->s', X, Sinv, X)        # (S,)
    g = torch.exp(-0.5 * q).reshape(size, size)
    g = g / (g.sum() + 1e-12)
    return g

def rotation_2x2(theta_rad: float) -> torch.Tensor:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return torch.tensor([[c, -s],
                         [s,  c]], dtype=torch.float32)   # (2,2), orthonormal

def make_fixed_U_list(K: int, r_spatial: int = 2):
    """
    Evenly-spaced orientations over [0, π).
    For r_spatial=2, returns K rotation matrices R(θ_k), shape (2,2) each.
    """
    assert r_spatial == 2, "For 2-D offsets, use r_spatial=2 (full basis) or adapt as needed."
    U_list = []
    for k in range(K):
        theta = math.pi * k / K  # K distinct axes
        U_list.append(rotation_2x2(theta))   # (2,2)
    return U_list

# ----------------------- spatial covariance with fixed U -----------------------

class SpatialCov2D_FixedU(nn.Module):
    """
    Σ = eps*I + U diag(alpha) U^T, with U fixed (orthonormal columns).
    alpha is learnable (nonnegative via softplus).
    """
    def __init__(self, U_fixed: torch.Tensor, eps: float = 1e-3):
        super().__init__()
        assert U_fixed.ndim == 2 and U_fixed.shape[0] == 2
        self.register_buffer('U', U_fixed.clone())          # (2, r_spatial), non-trainable
        self.alpha_raw = nn.Parameter(torch.zeros(U_fixed.shape[1]))  # (r_spatial,)
        self.eps = eps

    def Sigma(self, device=None, dtype=None) -> torch.Tensor:
        U = self.U.to(device=device, dtype=dtype)           # (2, r)
        alpha = F.softplus(self.alpha_raw).to(U)            # (r,)
        A = U * alpha                                       # scale each column by alpha_i
        return A @ U.T + self.eps * torch.eye(2, device=U.device, dtype=U.dtype)  # (2,2)

# ----------------------- bank of K Gaussian bands (fixed U_k) -----------------------

class GaussianBankFixedU(nn.Module):
    """
    Holds K bands, each with its own Σ_k = eps*I + U_k diag(alpha_k) U_k^T (U_k fixed),
    and applies each band as a depthwise Gaussian to the input feature map.

    forward(x): x is (C, H, W)  -> returns (K, C, H, W)
    where each slice [k] is the depthwise-convolved map with band k's kernel.
    """
    def __init__(self, K: int, size: int = 9, U_list=None, eps: float = 1e-3):
        super().__init__()
        assert size % 2 == 1
        self.K = K
        self.size = size
        if U_list is None:
            U_list = make_fixed_U_list(K, r_spatial=2)  # list of K (2,2) tensors
        assert len(U_list) == K
        self.covs = nn.ModuleList([SpatialCov2D_FixedU(U, eps=eps) for U in U_list])

    @staticmethod
    def _depthwise_conv(x: torch.Tensor, kernel_2d: torch.Tensor) -> torch.Tensor:
        # x: (C,H,W), kernel_2d: (h,w) -> depthwise conv with groups=C
        C, H, W = x.shape
        w = kernel_2d[None, None, ...].repeat(C, 1, 1, 1)   # (C,1,h,w)
        pad = kernel_2d.shape[-1] // 2
        # Add a batch dimension for conv2d, then remove it.
        return F.conv2d(circular_pad(x.unsqueeze(0), pad), w, groups=C).squeeze(0)  # (C,H,W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_per_band = []
        for k in range(self.K):
            Sigma_k = self.covs[k].Sigma(device=x.device, dtype=x.dtype)   # (2,2)
            gk = gaussian_kernel_2d_from_Sigma_chol(self.size, Sigma_k)         # (h,w)
            xk = self._depthwise_conv(x, gk)                               # (C,H,W)
            out_per_band.append(xk)
        return torch.stack(out_per_band, dim=0)                            # (K,C,H,W)


def gaussian_kernel_2d_from_Sigma_chol(size, Sigma, device=None, dtype=None):
    assert size % 2 == 1
    r = size // 2
    device = device or Sigma.device
    dtype  = dtype  or Sigma.dtype
    ys, xs = torch.meshgrid(
        torch.arange(-r, r+1, device=device, dtype=dtype),
        torch.arange(-r, r+1, device=device, dtype=dtype), indexing='ij'
    )
    X = torch.stack([xs, ys], dim=-1).reshape(-1, 2)       # (S,2), S=size^2

    # Cholesky factor (numerically stable): Sigma = L L^T, L lower-triangular
    L = torch.linalg.cholesky(Sigma)                       # (2,2)

    # Solve L Y = X^T  (triangular solve), then q = ||Y||^2 columnwise
    # This avoids forming Sigma^{-1} explicitly.
    Y = torch.linalg.solve_triangular(L, X.T, upper=False) # (2,S)
    q = (Y ** 2).sum(dim=0)                                # (S,)

    g = torch.exp(-0.5 * q).reshape(size, size)
    g = g / g.sum()
    return g
    
def gaussian_kernel_2d_from_Sigma(size, Sigma, device=None, dtype=None):
    assert size % 2 == 1
    r = size // 2
    device = device or Sigma.device
    dtype  = dtype  or Sigma.dtype
    ys, xs = torch.meshgrid(
        torch.arange(-r, r+1, device=device, dtype=dtype),
        torch.arange(-r, r+1, device=device, dtype=dtype), indexing='ij'
    )
    X = torch.stack([xs, ys], dim=-1).reshape(-1, 2)       # (S,2), S=size^2
    Sinv = torch.linalg.inv(Sigma)                         # (2,2)
    q = torch.einsum('sd,df,sf->s', X, Sinv, X)            # (S,)
    g = torch.exp(-0.5*q).reshape(size, size)              # (size,size)
    g = g / g.sum()
    return g

class ConvGaussian(nn.Module):
    def __init__(self,num_channels=4525, kernel_size=3, num_sub_features = 128, num_bands=10):
        super(ConvGaussian, self).__init__()
        self.num_sub_features = num_sub_features
        self.proj_q = torch.nn.Conv2d(num_channels, num_sub_features, kernel_size=1, bias=False)
        self.proj_k = torch.nn.Conv2d(num_channels, num_sub_features, kernel_size=1, bias=False)
        U_list = make_fixed_U_list(num_bands, r_spatial=2) # 4 rotations over [0, π)
        self.bank = GaussianBankFixedU(K=num_bands, size = int(kernel_size**2), U_list=U_list)

    def forward(self, x):
        q_all = self.proj_q(x)
        k_all = self.proj_k(x)
        k_s_all = self.bank(k_all)
        R = (q_all * k_s_all).sum(dim=1)/torch.sqrt(torch.tensor(self.num_sub_features).float())
        return R

class DoubleConvGaussian(nn.Module):
    def __init__(self,num_channels=4525, kernel_size=3, num_sub_features = 128, num_bands=10, num_classes=10, rank_A=5):
        super(DoubleConvGaussian, self).__init__()
        self.num_bands = num_bands
        self.conv_pos = ConvGaussian(num_channels, kernel_size, num_sub_features, num_bands)
        self.conv_neg = ConvGaussian(num_channels, kernel_size, num_sub_features, num_bands)
        self.A_pos = nn.Parameter(0.1+torch.zeros(num_bands, rank_A))
        self.b_pos = nn.Parameter(0.1+torch.zeros(num_bands))
        self.A_neg = nn.Parameter(0.1+torch.zeros(num_bands, rank_A))
        self.b_neg = nn.Parameter(0.1+torch.zeros(num_bands))
        self.tau_raw = nn.Parameter(torch.tensor(1.0))
        # self.linear_pos = nn.Linear(num_classes, num_classes)
        # self.linear_neg = nn.Linear(num_classes, num_classes)

    def lse(self,R):
        # R has shape (H, W)
        tau = F.softplus(self.tau_raw) + 1e-6              # (K,)
        # broadcast τ over spatial dims
        m = R.max()         # scalar
        Z = tau * (
                torch.log(torch.exp((R - m) / tau).sum())
                + m / tau
            )
        # Z is a scalar
        return Z

    def bilinear(self,x,positive_or_negative):
        if positive_or_negative == 'positive':
            A_pos = torch.einsum('ar,br->ab',self.A_pos, self.A_pos)
            b_pos = self.b_pos
            class_patch = torch.einsum('bb,bhw,bhw->hw',A_pos, x,x)+ torch.einsum('b,bhw->hw',b_pos,x)
            return class_patch
        elif positive_or_negative == 'negative':
            A_neg = torch.einsum('ar,br->ab',self.A_neg, self.A_neg)
            b_neg = self.b_neg
            class_patch = torch.einsum('bb,bhw,bhw->hw',A_neg, x,x)+ torch.einsum('b,bhw->hw',b_neg,x)
            return class_patch
    def forward(self,x):
        pos_x = self.conv_pos(x)
        neg_x = self.conv_neg(x)
        pos_patch = self.bilinear(pos_x,'positive')
        neg_patch = self.bilinear(neg_x,'negative')
        # pos_logits = self.linear_pos(pos_patch.T)
        # neg_logits = self.linear_neg(neg_patch.T)
        
        neg_lse = self.lse(neg_patch.T)
        pos_lse = self.lse(pos_patch.T)
        # print(neg_lse.shape,pos_lse.shape)
        # input('yipo')
        return torch.sigmoid(pos_lse)

class DoubleConvGaussianLabels(nn.Module):
    def __init__(self,num_channels=4525, kernel_size=3, num_sub_features = 128, num_bands=10, num_classes=10, rank_A=5):
        super(DoubleConvGaussianLabels, self).__init__()
        self.num_classes = num_classes
        self.gaussians = nn.ModuleList([DoubleConvGaussian(num_channels=num_channels, num_classes=12, kernel_size=3, num_sub_features = 256, num_bands=10) for _ in range(num_classes)])
    def forward(self,x):
        pred = []
        for i in range(self.num_classes):
            pred.append(self.gaussians[i](x))
        pred = torch.stack(pred, dim=0)
        return pred

class ChannelwiseSpatialMHSA(nn.Module):
    """
    Per-channel spatial multi-head self-attention.

    Input:  x  [B, H, W, C]
    Output:
        - if reduce == 'none' : [B, H, W, C, C_out]  (per-channel output kept)
        - else                : [B, H, W, C_out]     (channels merged after attention)

    Each channel c is processed independently: attention is computed over the S=H*W
    positions for that channel only (batched as B*C).
    """
    def __init__(self, c_in, c_out, d_model=64, num_heads=4, reduce='linear'):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.c_in = c_in
        self.c_out = c_out
        self.d_model = d_model
        self.h = num_heads
        self.d_head = d_model // num_heads
        self.reduce = reduce

        # Embed scalar per-location value to d_model tokens (shared across channels)
        self.embed = nn.Linear(1, d_model, bias=False)

        # Multi-head projections (applied on the d_model token)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Project concatenated heads to per-position, per-channel output dimension
        self.o_proj = nn.Linear(d_model, c_out, bias=False)

        # Optional channel merging after per-channel attention
        if reduce == 'linear':
            # Learn to merge the C per-channel outputs to 1 (shared across spatial positions)
            self.channel_merge = nn.Linear(c_in, 1, bias=False)
        elif reduce in ('sum', 'mean', 'none'):
            self.channel_merge = None
        else:
            raise ValueError("reduce must be one of {'none','linear','sum','mean'}")

        
    def forward(self, x):  # x: [B, H, W, C]
        B, H, W, C = x.shape
        assert C == self.c_in, f"expected C={self.c_in}, got {C}"
        S = H * W

        # Treat each channel independently in the batch: [B, H, W, C] -> [B*C, S, 1]
        t = x.permute(0, 3, 1, 2).contiguous().view(B * C, S, 1)

        # Token embedding per position
        E = self.embed(t)  # [B*C, S, d_model]

        # Multi-head projections
        Q = self.q_proj(E)  # [B*C, S, d_model]
        K = self.k_proj(E)  # [B*C, S, d_model]
        V = self.v_proj(E)  # [B*C, S, d_model]

        # Reshape to heads: [B*C, S, h, d_head] -> [B*C, h, S, d_head]
        def to_heads(T):
            return T.view(B * C, S, self.h, self.d_head).transpose(1, 2).contiguous()
        Qh, Kh, Vh = map(to_heads, (Q, K, V))  # each: [B*C, h, S, d_head]

        # Attention weights per channel/head over spatial positions
        attn_scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B*C, h, S, S]
        attn = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        Yh = torch.matmul(attn, Vh)  # [B*C, h, S, d_head]
        # Concatenate heads
        Y = Yh.transpose(1, 2).contiguous().view(B * C, S, self.d_model)  # [B*C, S, d_model]
        # Project to desired per-position output dim
        Y = self.o_proj(Y)  # [B*C, S, C_out]
        # Back to spatial and channel axes: [B, C, H, W, C_out]
        Y = Y.view(B, C, H, W, self.c_out)
        if self.reduce == 'none':
            # Return per-channel outputs
            # Apply LN over the last feature dim (C_out)
            Y = self.ln_out(Y)
            return Y  # [B, C, H, W, C_out] (caller can permute if needed)

        # Merge channels -> [B, H, W, C_out]
        if self.reduce == 'linear':
            # Linear combination over channel axis
            # channel_merge operates on last dim, so move C into last place temporarily
            Y_perm = Y.permute(0, 2, 3, 4, 1).contiguous()  # [B, H, W, C_out, C]
            Y_flat = Y_perm.view(B * H * W, self.c_out, C)  # [BHW, C_out, C]
            merged = self.channel_merge(Y_flat).squeeze(-1)  # [BHW, C_out]
            out = merged.view(B, H, W, self.c_out)
        elif self.reduce == 'sum':
            out = Y.sum(dim=1)  # sum over C -> [B, H, W, C_out]
            
        elif self.reduce == 'mean':
            out = Y.mean(dim=1)  # mean over C -> [B, H, W, C_out]
        else:
            raise RuntimeError("unreachable")
        return out

class SimpleModel(nn.Module):
    def __init__(self,in_channels=136,num_bands=64,num_classes=12,H_in=32,W_in=32,hidden_size=128):
        super(SimpleModel, self).__init__()
        self.proj_q = torch.nn.Conv2d(in_channels, num_bands, kernel_size=4, stride=1,padding=1, bias=False)
        self.proj_a = torch.nn.Conv2d(num_bands, num_bands//4, kernel_size=3, stride=1,padding='same', bias=False)
        self.tau_raw = nn.Parameter(torch.tensor(1.0))
        self.gaussian_kernels_patch = nn.Parameter(torch.ones(num_bands,1)*0.5)
        self.gaussian_kernels_classes = nn.Parameter(torch.ones(num_classes,1)*0.5)
        self.K = num_bands
        self.eps = 1e-4
        self.attn = nn.MultiheadAttention(embed_dim=num_bands, num_heads=1, batch_first=True)
        self.embed_in = nn.Linear(1, num_bands)
        self.embed_out = nn.Linear(num_bands, 1)
        self.attention_pool = ChannelwiseSpatialMHSA(num_bands//4,num_classes,reduce='sum',num_heads=2)
        self.layer_norm = nn.LayerNorm(num_classes)
        self.batch_norm = nn.BatchNorm2d(num_bands)
        self.linear_1 = nn.Linear(num_classes, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, num_classes)
    def patch_grid_conv(self,x):
        B,H_in,W_in,C,PH_in,PW_in = x.shape
        x = x.view(B*H_in*W_in,C,PH_in,PW_in)
        x = self.proj_q(x)
        x = self.batch_norm(x)
        PH_out = x.shape[-2]
        PW_out = x.shape[-1]
        x = x.view(B,H_in,W_in,self.K,PH_out,PW_out)
        return x

    def patch(self, x, patch_size, stride):
        B, C, H_in, W_in = x.shape
        flat = F.unfold(x, kernel_size=patch_size, stride=stride)  # (B, C*k*k, L)
        B, Ck2, L = flat.shape

        H_out = int((H_in - patch_size) / stride) + 1
        W_out = int((W_in - patch_size) / stride) + 1
        assert H_out * W_out == L, "patch shape mismatch"

        grid = flat.transpose(1, 2).contiguous()                  # (B, L, C*k*k)
        grid = grid.view(B, H_out, W_out, C, patch_size, patch_size)
        return grid
    
    def unpatch(self, x, patch_size, stride):
        B, H_p, W_p = x.shape          # x is [B, 32, 32] (per-class, per-patch scalar)

        x_scalar = x.unsqueeze(1)      # [B, 1, 32, 32]
        x_expanded = (
            x_scalar
            .repeat_interleave(patch_size, dim=2)
            .repeat_interleave(patch_size, dim=3)
            .squeeze(1)
        )
        return x_expanded              # [B, 512, 512]

    def lse(self,R):
        # R has shape (B, H, W) or (H, W)
        if R.ndim == 2:
            R = R.unsqueeze(0) # Treat as a batch of 1

        B, H, W = R.shape
        tau = F.softplus(self.tau_raw) + 1e-6              # (K,)
        
        # Find max over H, W for each item in the batch, result shape (B,)
        m = R.view(B, -1).max(dim=1)[0]
        
        # Subtract max for stability, broadcast m to (B, 1, 1)
        Z = tau * (
                torch.log(torch.exp((R - m.view(B, 1, 1)) / tau).view(B, -1).sum(dim=1))
                + m / tau
            )
        return Z.squeeze() # Squeeze in case B=1 to return a scalar
    
    @staticmethod
    def _kernel_from_sigma(sigma, ksize=15):   # returns (K,1,3,3), unit-sum per band
        """
        Generate a Gaussian kernel for each sigma in the batch.
        Returns a tensor of shape (K, 1, ksize, ksize)
        """
        K = sigma.shape[0]
        half = ksize // 2
        coords = torch.arange(-half, half + 1, device=sigma.device).float()  # (ksize,)
        g1d = torch.exp(-0.5 * (coords[None, :]**2) / (sigma[:, None]**2)).squeeze(1)   # (K, ksize)
        g1d = g1d / g1d.sum(dim=1, keepdim=True)  # normalize 1D
        # Outer product to make 2D kernel
        g2d = torch.einsum('ki,kj->kij', g1d, g1d)  # (K, ksize, ksize)
        g2d = g2d / g2d.view(K, -1).sum(dim=1, keepdim=True).view(-1, 1, 1)  # normalize 2D

        return g2d.unsqueeze(1)  # shape: (K, 1, ksize, ksize)
    def forward(self,x):
        x = self.patch_grid_conv(x)
        B,H_in,W_in,C,PH_in,PW_in = x.shape
        sigma_patch = torch.nn.functional.softplus(self.gaussian_kernels_patch) + self.eps  # (K,)
        weight_patch = self._kernel_from_sigma(sigma_patch,ksize=PH_in).to(x.dtype).to(x.device)
        B,H_in,W_in,C,PH_in,PW_in = x.shape
        x = x.view(B*H_in*W_in,C,PH_in,PW_in)
        x = F.conv2d(x, weight=weight_patch, bias=None, stride=1, padding=1, groups=self.K)
        B_patch,C,PH_out,PW_out = x.shape
        x = x.view(B_patch*C,PH_out,PW_out)
        x = self.lse(x)
        x = x.view(B,H_in,W_in,C).permute(0,3,1,2)
        x = self.proj_a(x)
        
        x = x.permute(0,2,3,1)
        x = self.attention_pool(x)
        return x
    
    def inequality_eval(self,x):
        return torch.nn.functional.softplus(x)

    def normalize_prediction(self,x):
        return (x - x.mean(dim=(1,2),keepdim=True))/x.std(dim=(1,2),keepdim=True)

    def constraint_pos_geq(self,x):
        greater_than = torch.where(x>0)
        mean = x[greater_than].mean()
        std = x[greater_than].std()
        const_geq = -(mean-2*std)
        return const_geq, mean,std
    def constraint_neg_leq(self,x):
        less_than = torch.where(x<0)
        mean = x[less_than].mean()
        std = x[less_than].std()
        const_leq = mean+2*std
        return const_leq, mean,std
    def eval_constraints(self,x):

        x_normalized = self.normalize_prediction(x)
        less_than = torch.where(x_normalized<0)
        greater_than = torch.where(x_normalized>0)
        if torch.any(less_than[0]).item() and torch.any(greater_than[0]).item():

            const_geq,mean_geq,std_geq = self.constraint_pos_geq(x_normalized)
            const_leq,mean_leq,std_leq = self.constraint_neg_leq(x_normalized)
            constraints = torch.concatenate((const_geq.unsqueeze(0),const_leq.unsqueeze(0)),dim=0)
            constraints = torch.where(constraints>0.1,constraints,0)
            counter = 0
            while torch.any(constraints>0.1).item():
                constraints = self.inequality_eval(constraints)
                grad = torch.autograd.grad(constraints.sum(),x,create_graph=True,retain_graph=True)[0]
                x = x - 1000*grad
                x_normalized = self.normalize_prediction(x)
                const_geq,mean_geq,std_geq = self.constraint_pos_geq(x_normalized)
                const_leq,mean_leq,std_leq = self.constraint_neg_leq(x_normalized)
                constraints = torch.concatenate((const_geq.unsqueeze(0),const_leq.unsqueeze(0)),dim=0)
                constraints = torch.where(constraints>0.1,constraints,0)
                print("gradient step",counter,constraints)
                counter += 1
                if counter>100:
                    break
        return self.normalize_prediction(x)

import math

class PatchGraphModelNoBatch(nn.Module):
    """
    Input x: (L=256, C=136, H=32, W=32)  # 256 patches, already patchified
    Steps:
      1) 1x1 conv -> K bands (per patch, per pixel)
      2) depthwise Gaussian blur per band (learnable sigma_k), no band-sum
      3) Pool *within each patch* (over 32x32) to get one K-dim token per patch
         (normalized LSE; swap for mean if preferred)
      4) Single-head self-attention across the 256 tokens (length-preserving)
      5) Linear head -> one logit per patch (256,)
    """
    def __init__(self, in_channels=62, num_bands=128, patch=32):
        super().__init__()
        self.K = num_bands
        self.patch = patch
        self.eps = 1e-4

        # 1) per-pixel projection to K bands
        self.proj_q = nn.Conv2d(in_channels, num_bands, kernel_size=1, bias=False)

        # 2) per-band Gaussian spread: sigma_k = softplus(rho_k) + eps
        self.rho = nn.Parameter(torch.zeros(num_bands))

        # pooling temperature (mean <-> max bridge)
        self.tau_raw = nn.Parameter(torch.tensor(1.0))

        # 4) attention across patches (tokens are K-dim). One head.
        self.attn = nn.MultiheadAttention(embed_dim=num_bands, num_heads=1, batch_first=True)
        self.ln1  = nn.LayerNorm(num_bands)
        self.ffn  = nn.Sequential(
            nn.Linear(num_bands, 4 * num_bands), nn.GELU(),
            nn.Linear(4 * num_bands, num_bands)
        )
        self.ln2  = nn.LayerNorm(num_bands)

        # 5) scalar head per patch
        self.head = nn.Linear(num_bands, 1)

    @staticmethod
    def _gauss3x3_from_sigma(sig):  # (K,) -> (K,1,3,3), unit-sum
        e = torch.exp(-0.5 / (sig**2))                      # (K,)
        g = torch.stack([e, torch.ones_like(e), e], dim=1)  # (K,3)
        g = g / g.sum(dim=1, keepdim=True)
        K2 = torch.einsum('ki,kj->kij', g, g)               # (K,3,3)
        K2 = K2 / K2.view(K2.size(0), -1).sum(dim=1, keepdim=True).view(-1,1,1)
        return K2.unsqueeze(1)                               # (K,1,3,3)

    def _depthwise_gauss(self, x):   # x: (L,K,H,W) -> (L,K,H,W)
        sigma = F.softplus(self.rho) + self.eps             # (K,)
        W = self._gauss3x3_from_sigma(sigma).to(x.device, x.dtype)
        # reflect padding avoids edge darkening inside each 32x32 patch
        x_pad = F.pad(x, (1,1,1,1), mode='reflect')
        return F.conv2d(x_pad, W, stride=1, padding=0, groups=self.K)

    @staticmethod
    def _lse_normalized_lastdim(z, tau):
        """
        z: (..., P)  -- LSE over last dim with -log P normalization
        returns: (...,)
        """
        P = z.size(-1)
        u = z / tau
        m = u.max(dim=-1, keepdim=True).values
        return tau * (torch.log(torch.exp(u - m).sum(dim=-1)) + m.squeeze(-1) - math.log(P))

    def forward(self, x):  # x: (L=256, C=136, H=32, W=32)
        L, C, H, W = x.shape
        assert H == self.patch and W == self.patch, "Expecting pre-cut 32x32 patches"

        # 1) -> (L,K,32,32)
        x = self.proj_q(x)

        # 2) per-band Gaussian blur (no band sum)
        x = self._depthwise_gauss(x)                        # (L,K,32,32)

        # 3) pool *within each patch* over spatial dims to get one K-dim token per patch
        #    (use normalized LSE; swap to x.mean(dim=(2,3)) for simple mean)
        tau = F.softplus(self.tau_raw) + 1e-6
        z = x.view(L, self.K, -1)                           # (L,K,P) with P=1024
        tokens = self._lse_normalized_lastdim(z, tau)       # (L,K)

        # 4) single-head self-attention *across patches only*
        #    attn expects (B, L, E); we add a dummy batch of 1 and then remove it
        seq = tokens.unsqueeze(0)                           # (1, L, K)
        y, _ = self.attn(seq, seq, seq)                    # (1, L, K)
        y = self.ln1(seq + y)
        y2 = self.ffn(y)
        y = self.ln2(y + y2)                               # (1, L, K)
        y = y.squeeze(0)                                    # (L, K)  order preserved

        # 5) one logit per patch
        logits = self.head(y).squeeze(-1)                  # (L,)

        return torch.sigmoid(logits)  # use BCEWithLogitsLoss; apply sigmoid only for inference
    



class Centroid(nn.Module):
    def __init__(self, in_channels:int =3, eps: float = 1e-3):
        super().__init__()
        self.mu = nn.Parameter(torch.ones(in_channels)*0.5,requires_grad=True)
        self.L_raw = nn.Parameter(torch.tril(torch.randn(in_channels,in_channels)),requires_grad=True)
    def make_L(self, min_diag=1e-4):
        L = torch.tril(self.L_raw)  # zero-out above-diagonal
        diag = torch.nn.functional.softplus(torch.diagonal(L, 0)) + min_diag
        L = L.clone()
        L.diagonal(0).copy_(diag)
        return L
    def make_Sigma(self, eps=1e-4):
        L = torch.tril(self.make_L())
        diag = torch.nn.functional.softplus(torch.diagonal(L, 0)) + eps
        L = L.clone()
        L.diagonal(0).copy_(diag)
        Sigma = L @ L.T + eps*torch.eye(L.shape[0],device=L.device,dtype=L.dtype)
        return Sigma
    def forward(self,x,pi):
        sigma = self.make_Sigma()
        diff = (x-self.mu)
        jitter = 1e-3
        likelihood = (diff@torch.linalg.pinv(sigma+jitter*torch.eye(sigma.shape[0],device=sigma.device,dtype=sigma.dtype))*diff).sum(-1)
        determinant = torch.linalg.det(sigma)
        return (-likelihood/2-torch.log(determinant)/2+torch.log(pi))

class GMM(nn.Module):
    def __init__(self, in_channels:int =3, eps: float = 1e-3, num_gaussians=5):
        super().__init__()
        self.gaussians = nn.ModuleList([Centroid(in_channels,eps) for _ in range(num_gaussians)])
        self.pi = nn.Parameter(torch.ones(num_gaussians)/num_gaussians,requires_grad=True)
    def logits(self,x):
        pi = torch.softmax(self.pi,dim=0)
        return torch.stack([self.gaussians[i](x,pi[i]) for i in range(len(self.gaussians))],dim=1)
    def forward(self,x):
        logits = self.logits(x)
        responsibilities = torch.softmax(logits,dim=1)
        return responsibilities
    def forward_loss(self,x):
        logits = self.logits(x)
        responsibilities = self.forward(x)
        entropy = torch.special.xlogy(responsibilities,responsibilities)
        return -responsibilities*logits+entropy

class AsymmetricGaussianMF(nn.Module):
    def __init__(self,num_mfs=3, c_init=0.0, sigmaL_init=0.05, sigmaR_init=0.05, min_sigma=1e-3, device='cpu'):
        super().__init__()
        c= torch.linspace(0,1,num_mfs)
        self.c = nn.Parameter(torch.ones(num_mfs,requires_grad=True,dtype=torch.float32,device=device)*c)
        # store raw params; map to positive via softplus in forward
        self.sigmaL_raw = nn.Parameter(torch.ones(num_mfs,requires_grad=True,dtype=torch.float32,device=device)*sigmaL_init)
        self.sigmaR_raw = nn.Parameter(torch.ones(num_mfs,requires_grad=True,dtype=torch.float32,device=device)*sigmaR_init)
        self.min_sigma = min_sigma

    def forward(self,x, eps=1e-12,T=1.0):
        """
        x: [N] or [N,1]
        c, sigma_L, sigma_R: [M]
        returns memberships: [N, M]
        """
        x = x.unsqueeze(-1)              # [N,1]
        sL = F.softplus(self.sigmaL_raw) + eps   # [M]
        sR = F.softplus(self.sigmaR_raw) + eps   # [M]
        # broadcast: compare x vs c per MF
        sigma = torch.where(x < self.c, sL, sR)   # [N,M]
        z = (x - self.c) / sigma
        z = -0.5 * z * z
        return F.softmax(z/T, dim=1)

class GaussianDownsample2x(nn.Module):
    """
    Anti-aliased 2× downsampling via depthwise Gaussian conv2d (stride=2).
    - Input  : (N, C, H, W)
    - Output : (N, C, H/2, W/2)   (H, W must be even or padding will extend)
    Args:
        sigma (float): Gaussian sigma in pixels.
        kernel_size (int or None): odd kernel size. If None, computed as 2*round(3*sigma)+1.
        padding_mode (str): 'reflect' (recommended), 'replicate', or 'zeros'.
    """
    def __init__(self, sigma: float = 1.0, kernel_size: int = None, padding_mode: str = 'reflect'):
        super().__init__()
        assert sigma > 0, "sigma must be > 0"
        if kernel_size is None:
            # cover ±3σ
            kernel_size = int(2 * round(3 * sigma) + 1)
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.sigma = float(sigma)
        self.kernel_size = int(kernel_size)
        self.padding_mode = padding_mode

        # Build normalized 1D Gaussian kernel
        radius = self.kernel_size // 2
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        g1d = torch.exp(-(x ** 2) / (2 * self.sigma ** 2))
        g1d = g1d / g1d.sum()

        # 2D separable kernel via outer product
        g2d = torch.outer(g1d, g1d)  # (k, k)
        g2d = g2d / g2d.sum()        # just to be safe

        # Register as buffer; we’ll expand to (C,1,k,k) at runtime
        self.register_buffer('_kernel2d', g2d[None, None])  # shape (1,1,k,k)

        # Cache to avoid re-expanding every forward
        self._cached_C = None
        self.register_buffer('_weight', None)  # will hold (C,1,k,k)

    @torch.no_grad()
    def _get_weight(self, C: int, device, dtype):
        """
        Expand the (1,1,k,k) kernel to (C,1,k,k) on the right device/dtype.
        Cache per-channel count to avoid repeated expands.
        """
        if (self._cached_C != C) or (self._weight is None) \
           or (self._weight.device != device) or (self._weight.dtype != dtype):
            w = self._kernel2d.to(device=device, dtype=dtype).expand(C, 1, self.kernel_size, self.kernel_size).contiguous()
            self._weight = w
            self._cached_C = C
        return self._weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur then downsample by stride=2 (single conv).
        """
        assert x.dim() == 4, "Expected input of shape (N, C, H, W)"
        N, C, H, W = x.shape

        pad = self.kernel_size // 2
        if self.padding_mode == 'reflect':
            x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        elif self.padding_mode == 'replicate':
            x = F.pad(x, (pad, pad, pad, pad), mode='replicate')
        elif self.padding_mode == 'zeros':
            x = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0.0)
        else:
            raise ValueError(f"Unknown padding_mode: {self.padding_mode}")

        w = self._get_weight(C, x.device, x.dtype)  # (C,1,k,k)
        # Depthwise conv: groups=C makes each channel filtered independently
        y = F.conv2d(x, w, bias=None, stride=2, padding=0, groups=C)
        return y


class GaussianUpsample2x(nn.Module):
    """
    Approximate synthesis for GaussianDownsample2x:
      - Depthwise transposed conv (stride=2) with the same Gaussian kernel
      - Kernel scaled by 4 to preserve DC after 2x decimation
      - Crops 'pad' pixels to undo the forward external padding
      - If out_hw is None: outputs exactly (2*Hs, 2*Ws)
      - Optional mild unsharp-mask
    """
    def __init__(self, sigma: float = 1.0, kernel_size: int = None, padding_mode: str = 'reflect',
                 sharpen_amount: float = 0.0):
        super().__init__()
        assert sigma > 0
        if kernel_size is None:
            kernel_size = int(2 * round(3 * sigma) + 1)  # odd
        assert kernel_size % 2 == 1
        self.sigma = float(sigma)
        self.kernel_size = int(kernel_size)
        self.padding_mode = padding_mode
        self.sharpen_amount = float(sharpen_amount)

        # Build normalized Gaussian
        radius = self.kernel_size // 2
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        g1d = torch.exp(-(x ** 2) / (2 * self.sigma ** 2))
        g1d = g1d / g1d.sum()
        g2d = torch.outer(g1d, g1d)
        g2d = g2d / g2d.sum()

        self.register_buffer('_kernel2d', g2d[None, None])  # (1,1,k,k)

        self._cached_C = None
        self.register_buffer('_w_t', None)  # (C,1,k,k) for transpose
        self.register_buffer('_w_f', None)  # (C,1,k,k) for optional blur

    @torch.no_grad()
    def _get_weights(self, C: int, device, dtype):
        if (self._cached_C != C) or (self._w_t is None) \
           or (self._w_t.device != device) or (self._w_t.dtype != dtype):
            k = self._kernel2d.to(device=device, dtype=dtype).expand(
                C, 1, self.kernel_size, self.kernel_size
            ).contiguous()
            self._w_t = (4.0 * k)  # synthesis gain to restore DC after decimation
            self._w_f = k
            self._cached_C = C
        return self._w_t, self._w_f

    def forward(self, y: torch.Tensor, out_hw=None) -> torch.Tensor:
        """
        y: (N, C, Hs, Ws)
        returns:
          - if out_hw is None: shape (N, C, 2*Hs, 2*Ws)
          - else: (N, C, H, W) with exact (H,W)
        """
        assert y.dim() == 4, "Expected (N, C, Hs, Ws)"
        N, C, Hs, Ws = y.shape
        w_t, w_f = self._get_weights(C, y.device, y.dtype)

        p = self.kernel_size // 2

        if out_hw is None:
            # Target exact doubling: (2*Hs, 2*Ws)
            opad_h = 1  # because (2*Hs - 1) + 1 = 2*Hs after cropping p each side
            opad_w = 1
            x = F.conv_transpose2d(
                y, w_t, bias=None, stride=2, padding=0,
                output_padding=(opad_h, opad_w),
                groups=C
            )
            # Undo the forward external pad by cropping p each side
            if p > 0:
                x = x[..., p:-p, p:-p]
            # Final clamp to exact 2x in case of border quirks
            x = x[..., : (2*Hs), : (2*Ws)]
        else:
            # Target explicit (H, W)
            H, W = out_hw
            # Size before crop is (2*Hs - 1 + opad_h, 2*Ws - 1 + opad_w)
            # Solve for opad_* in {0,1} to match H,W after crop:
            # (2*Hs - 1 + opad_h) - 2*p  -> then clamp to H
            opad_h = int(max(0, min(1, H - (2*Hs - 1))))
            opad_w = int(max(0, min(1, W - (2*Ws - 1))))
            x = F.conv_transpose2d(
                y, w_t, bias=None, stride=2, padding=0,
                output_padding=(opad_h, opad_w),
                groups=C
            )
            if p > 0:
                x = x[..., p:-p, p:-p]
            x = x[..., :H, :W]

        if self.sharpen_amount > 0.0:
            # mild unsharp mask
            xb = F.pad(x, (p, p, p, p), mode=self.padding_mode) if p > 0 else x
            xb = F.conv2d(xb, w_f, bias=None, stride=1, padding=0, groups=C)
            x = x + self.sharpen_amount * (x - xb)

        return x


class FuzzyModel(nn.Module):
    def __init__(self, GMM,num_degrees=3, eps: float = 1e-3,kernel_size=7,num_convs=6, sub_patch_size=4,embed_dim=64,num_heads=8):
        super().__init__()
        self.gmm = GMM
        self.gaussian_memberships = nn.ModuleList([AsymmetricGaussianMF(num_mfs=num_degrees) for i in range(len(self.gmm.gaussians))])
        self.texture_memberships = AsymmetricGaussianMF(num_mfs=num_degrees)
        self.gaussian_downsample = GaussianDownsample2x(sigma=0.9, kernel_size=None, padding_mode='reflect')
        self.gaussian_upsample = GaussianUpsample2x(sigma=0.9, kernel_size=None, padding_mode='reflect')
        self.mha_intra_patch = nn.ModuleList([nn.MultiheadAttention(embed_dim=(len(self.gmm.gaussians)+1), num_heads=(len(self.gmm.gaussians)+1), batch_first=True,dropout=0.1) for _ in range(num_convs+1)])
        self.mha_inter_patch = nn.ModuleList([nn.MultiheadAttention(embed_dim=(len(self.gmm.gaussians)+1), num_heads=(len(self.gmm.gaussians)+1), batch_first=True,dropout=0.1) for _ in range(num_convs+1)])

        self.kernel_size = kernel_size
        self.num_convs = num_convs
        self.sub_patch_size = sub_patch_size
                
    def fuzzify(self,x_gaussian,x_sobel):
        N,H,W,C = x_gaussian.shape
        results = []
        for gaussian_mf in range(len(self.gaussian_memberships)):
            results.append(self.gaussian_memberships[gaussian_mf](x_gaussian[:,gaussian_mf]))
        results_gaussian = torch.stack(results,dim=2).permute(0,2,1)
        results_sobel = self.texture_memberships(x_sobel)
        results = torch.cat([results_gaussian,results_sobel],dim=1)
        _,F,M=results.shape
        return results.reshape(N,H,W,F*M)

    def centroid(self,x):
        x_gaussian = x[:,:,:,0:3]
        N,H,W,C = x_gaussian.shape
        x_gaussian = x_gaussian.reshape(N*H*W,C)
        out = self.gmm(x_gaussian)
        return out.reshape(N,H,W,-1)

    def patchify(self,x):
        N,C,H,W = x.shape
        gh, gw = H // self.sub_patch_size, W // self.sub_patch_size
        reshaped= x.reshape(N,C,gh,self.sub_patch_size,gw,self.sub_patch_size).permute(0,1,2,4,3,5).reshape(N,C,gh,gw,self.sub_patch_size*self.sub_patch_size)
        return reshaped
    
    
    def mha_via_sdpa(self, mha: torch.nn.MultiheadAttention, x: torch.Tensor,
                    attn_mask=None, key_padding_mask=None, is_causal=False):
        """
        x: (N, T, E) if mha.batch_first=True, else (T, N, E)
        Returns: y with same shape as x.
        Uses mha's parameters (in_proj + out_proj) but computes attention via SDPA.
        """
        assert mha.batch_first, "This helper assumes batch_first=True for simplicity."
        N, T, E = x.shape
        h = mha.num_heads
        assert E == mha.embed_dim
        d = E // h
        assert d * h == E

        # 1) Compute Q, K, V using the same packed projection as nn.MultiheadAttention
        #    qkv: (N, T, 3E)
        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)  # each: (N, T, E)

        # 2) Reshape to (N, h, T, d) as expected by SDPA
        q = q.view(N, T, h, d).transpose(1, 2)  # (N, h, T, d)
        k = k.view(N, T, h, d).transpose(1, 2)
        v = v.view(N, T, h, d).transpose(1, 2)

        # 3) Key padding mask handling (optional)
        # SDPA supports an attn_mask; for key padding mask, you can fold it into attn_mask.
        # key_padding_mask: (N, T) with True for "pad" positions to be masked out.
        if key_padding_mask is not None:
            # Build an additive mask with -inf on padded keys.
            # Shape should broadcast to (N, h, T, T): mask over keys dimension.
            # mask[n, 1, 1, s] = -inf if key_padding_mask[n, s] is True
            pad = key_padding_mask[:, None, None, :].to(torch.bool)  # (N,1,1,T)
            pad_mask = torch.zeros((N, 1, 1, T), device=x.device, dtype=x.dtype)
            pad_mask = pad_mask.masked_fill(pad, float("-inf"))
            attn_mask = pad_mask if attn_mask is None else attn_mask + pad_mask

        # 4) SDPA
        # Returns (N, h, T, d)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

        # 5) Merge heads: (N, T, E)
        y = y.transpose(1, 2).contiguous().view(N, T, E)

        # 6) Output projection (same as MHA)
        y = F.linear(y, mha.out_proj.weight, mha.out_proj.bias)
        return y


    def attention(self,x,conv_index,mha_list):

        B, C, H, W, T = x.shape
        # (B, C, H, W, T) -> (B, H, W, T, C)
        x_bhw_tc = x.permute(0, 2, 3, 4, 1)

        # Flatten spatial -> (B*H*W, T, C)
        x_flat = x_bhw_tc.reshape(B*H*W, T, C)
        
        y_flat = torch.empty_like(x_flat)
        chunk_size = 256
        for i in range(0,x_flat.shape[0],chunk_size):
            e = min(i+chunk_size,x_flat.shape[0])
            y_flat[i:e]=self.mha_via_sdpa(mha_list[conv_index],x_flat[i:e])

        # Restore shape -> (B, H, W, T, C)
        y_bhw_tc = y_flat.reshape(B, H, W, T, C)
        
        # Back to original layout if you prefer channels-first with time last: (B, C, H, W, T)
        y = y_bhw_tc.permute(0, 4, 1, 2, 3)
        
        return y
    def depatch(self,x):
        N,C,gH,gW,T = x.shape
        p = self.sub_patch_size
        assert p*p == T
        x = x.reshape(N,C,gH,gW,p,p).permute(0,1,2,4,3,5).reshape(N,C,p*gH,p*gW)
        return x

    def forward(self,x):
        
        x_sobel = x[:,:,:,3:]
        N,H,W,C=x.shape
        x_centroids = self.centroid(x)
        x = torch.concat([x_centroids,x_sobel],dim=3).permute(0,3,1,2)
        x_patch = self.patchify(x)
        x_intra_patch = self.attention(x_patch,0,self.mha_intra_patch)
        x_intra = self.depatch(x_intra_patch)
        x_patch_mean = x_patch.mean(dim=-1)
        x_patch_mean_patch = self.patchify(x_patch_mean)
        x_patch_mean_patch_inter = self.attention(x_patch_mean_patch,0,self.mha_inter_patch)
        x_inter = self.depatch(x_patch_mean_patch_inter)
        original_intras = [x_intra]
        original_inters = [x_inter]
        
        downsamples = [self.gaussian_downsample(original_intras[0])]
        upsample_intra = [original_intras[0]]
        upsample_inter = [self.gaussian_upsample(self.gaussian_upsample(original_inters[0]))]
        
        for i in range(self.num_convs-1):
            new_x = downsamples[-1]
            new_x_patch = self.patchify(new_x)
            new_x_intra_patch = self.attention(new_x_patch,i+1,self.mha_intra_patch)
            new_x_intra = self.depatch(new_x_intra_patch)
            original_intras.append(new_x_intra)
            new_x_patch_mean = new_x_patch.mean(dim=-1)
            new_x_patch_mean_patch = self.patchify(new_x_patch_mean)
            new_x_patch_mean_patch_inter = self.attention(new_x_patch_mean_patch,i+1,self.mha_inter_patch)
            new_x_inter = self.depatch(new_x_patch_mean_patch_inter)
            original_inters.append(new_x_inter)
            downsamples.append(self.gaussian_downsample(original_intras[-1]))
        for i in range(1,len(downsamples)):
            intra = original_intras[i]
            intra_upsamples = int(np.log2(original_intras[0].shape[-1]//intra.shape[-1]))
            inter = original_inters[i]
            inter_upsamples = int(np.log2((original_intras[0].shape[-1]//inter.shape[-1])))
            for _ in range(intra_upsamples):
                intra = self.gaussian_upsample(intra)
            upsample_intra.append(intra)
            for _ in range(inter_upsamples):
                inter = self.gaussian_upsample(inter)
            upsample_inter.append(inter)
            
        upsample_inter = torch.stack(upsample_inter,dim=4)
        upsample_intra = torch.stack(upsample_intra,dim=4)
        print(upsample_inter.shape,upsample_intra.shape)
        input('yipo')
        #     gaussians_same_size.append(upsampled)
        # gaussians_downsampled_upsampled = torch.stack(gaussians_same_size,dim=4)
        # print(gaussians_downsampled_upsampled.shape)
        # input('yipo')
        
        # input('yipo')