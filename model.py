import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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