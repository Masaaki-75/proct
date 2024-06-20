import math
import einops as E
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
import sys
sys.path.append('..')
from modules.basic_layers import get_conv_block, get_norm_layer, get_act_layer, DropPath
from modules.rope import RotaryEmbedding, apply_rot_embed
from utilities.initialization import init_weights_with_scale
from utilities.misc import ensure_tuple_rep
AVAIL_FAST_ATTN = False #hasattr(F, 'scaled_dot_product_attention')


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.GELU(),
            nn.Conv2d(hidden_features, out_features, 1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1/4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)
    
    
def vmap(module, x: torch.Tensor, *args, **kwargs):
    batch_size, support_size = x.shape[:2]
    in_shots = E.rearrange(x, "B S ... -> (B S) ...")
    out_shots = module(in_shots, *args, **kwargs)
    return E.rearrange(out_shots, "(B S) ... -> B S ...", B=batch_size, S=support_size)


class CrossConv2d(nn.Conv2d):
    """Adapted from UniverSeg, thanks!
    Parameters
    ----------
    in_channels : int or tuple of ints
        Number of channels in the input tensor(s).
        If the tensors have different number of channels, in_channels must be a tuple
    out_channels : int
        Number of output channels.
    Returns
    -------
    torch.Tensor
        Tensor resulting from the pairwise convolution between the elements of x and y.
    Examples
    --------
    >>> x = torch.randn(2, 3, 4, 32, 32)
    >>> y = torch.randn(2, 5, 6, 32, 32)
    >>> conv = CrossConv2d(in_channels=(4, 6), out_channels=7, kernel_size=3, padding=1)
    >>> output = conv(x, y)
    >>> output.shape  #(2, 3, 5, 7, 32, 32)
    """

    def __init__(
        self,  in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, padding_mode="zeros"):

        if isinstance(in_channels, (list, tuple)):
            concat_channels = sum(in_channels)
        else:
            concat_channels = 2 * in_channels
        self.concat_channels = concat_channels
        
        super().__init__(
            in_channels=concat_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        B, Sx, *_ = x.shape  # Sx should be 1 for only one input query
        _, Sy, *_ = y.shape

        xs = E.repeat(x, "B Sx Cx H W -> B Sx Sy Cx H W", Sy=Sy)
        ys = E.repeat(y, "B Sy Cy H W -> B Sx Sy Cy H W", Sx=Sx)
        xy = torch.cat([xs, ys], dim=3,)  # channels addition
        
        batched_xy = E.rearrange(xy, "B Sx Sy C2 H W -> (B Sx Sy) C2 H W")
        batched_output = super().forward(batched_xy)
        output = E.rearrange(batched_output, "(B Sx Sy) Co H W -> B Sx Sy Co H W", B=B, Sx=Sx, Sy=Sy)
        return output


class CrossFourierConv2d(nn.Module):
    def __init__(
        self,  in_channels, out_channels, stride=1, padding=0, dilation=1, groups=1, 
        padding_mode="reflect", norm_type=None, norm_kwargs=None, act_type='RELU', act_kwargs=None, 
        fft_type='ortho', reduction=2,):
        super().__init__()
        mid_channels = out_channels // reduction
        ffc_in_channels = ffc_out_channels = 2 * mid_channels
        conv_kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)
        block_kwargs = dict(norm_type=norm_type, norm_kwargs=norm_kwargs, act_type=act_type, act_kwargs=act_kwargs)
        self.fft_type = fft_type
        
        concat_channels = sum(in_channels) if isinstance(in_channels, (list, tuple)) else 2 * in_channels
        
        self.conv_in = get_conv_block(
            concat_channels, mid_channels, kernel_size=1, bias=False, adn_order='CNA', 
            **block_kwargs, **conv_kwargs)
        self.conv = nn.Conv2d(ffc_in_channels, ffc_out_channels, kernel_size=1, **conv_kwargs)
        self.norm = get_norm_layer(norm_type=norm_type, args=[ffc_out_channels], kwargs=norm_kwargs)
        self.act = get_act_layer(act_type=act_type, kwargs=act_kwargs)
        self.conv_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False, **conv_kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        B, Sx, *_ = x.shape  # Sx should be 1 for only one input query
        _, Sy, *_ = y.shape
        
        xs = E.repeat(x, "B Sx Cx H W -> B Sx Sy Cx H W", Sy=Sy)
        ys = E.repeat(y, "B Sy Cy H W -> B Sx Sy Cy H W", Sx=Sx)
        xy = torch.cat([xs, ys], dim=3,)
        
        batched_xy = E.rearrange(xy, "B Sx Sy C2 H W -> (B Sx Sy) C2 H W")  
        #eff_batch_size = batched_xy.shape[0]
        batched_xy = self.conv_in(batched_xy)  # (Be, Sx, Sy, Cm, H, W)
        ffted = torch.fft.rfftn(batched_xy, dim=(-2, -1), norm=self.fft_type)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)  # (Be, Cm, H, W/2+1, 2)
        ffted = E.rearrange(ffted, "B C H W L -> B (C L) H W")  # (Be, 2Cm, H, W/2+1)

        ffted = self.conv(ffted)  # (Be, 2Cm, H, W/2+1)
        ffted = self.act(self.norm(ffted))  # (Be, 2Cm, H, W/2+1)
        ffted = E.rearrange(ffted, "B (C L) H W -> B C H W L", L=2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        
        ifft_shape_slice = batched_xy.shape[2:]
        batched_iffted = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=(-2, -1), norm=self.fft_type)
        
        batched_out = self.conv_out(batched_xy + batched_iffted)

        output = E.rearrange(batched_out, "(B Sx Sy) Co H W -> B Sx Sy Co H W", B=B, Sx=Sx, Sy=Sy)
        return output
    

class CrossConvBlock(nn.Module):
    def __init__(
        self, in_channels, cross_channels, out_channels=None, kernel_size=3, stride=1, padding=1,
        norm_type=None, norm_kwargs=None, act_type="ReLU", act_kwargs=None, padding_mode='reflect', use_spectral=False):
        super().__init__()
        out_channels = cross_channels if out_channels is None else out_channels
        conv_kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        block_kwargs = dict(norm_type=norm_type, norm_kwargs=norm_kwargs, act_type=act_type, act_kwargs=act_kwargs)
        self.cross = CrossConv2d(in_channels=ensure_tuple_rep(in_channels, 2), out_channels=cross_channels, **conv_kwargs)
        self.norm = get_norm_layer(norm_type, args=out_channels, kwargs=norm_kwargs)
        self.act = get_act_layer(act_type, kwargs=act_kwargs)
        
        if use_spectral:
            freq_kwargs = dict(fft_type='ortho', reduction=2)
            self.cross_freq = CrossFourierConv2d(
                in_channels=ensure_tuple_rep(in_channels, 2), out_channels=cross_channels, 
                **block_kwargs, **freq_kwargs)
            self.norm_freq = get_norm_layer(norm_type, args=out_channels, kwargs=norm_kwargs)
            self.act_freq = get_act_layer(act_type, kwargs=act_kwargs)
            init_weights_with_scale([self.cross_freq, self.norm_freq], 1)
        else:
            self.cross_freq = self.act_freq = self.norm_freq = None
        
        self.query_branch = get_conv_block(cross_channels, out_channels, **block_kwargs, **conv_kwargs)
        self.context_branch = get_conv_block(cross_channels, out_channels, **block_kwargs, **conv_kwargs)
        init_weights_with_scale([self.cross, self.norm, self.query_branch, self.context_branch], 1)
        
    def forward(self, query, context):
        # [B, 1, Ciq, H, W], [B, S, Cic, H, W] -> [B, 1, Co, H, W], [B, S, Co, H, W]
        interaction = self.cross(query, context).squeeze(dim=1)  # squeeze Sx == 1
        interaction = vmap(self.norm, interaction)
        interaction = self.act(interaction)
        query_new = interaction.mean(dim=1, keepdims=True)
        
        if self.cross_freq is not None:
            interaction2 = self.cross_freq(query, context).squeeze(dim=1)  # squeeze Sx == 1
            interaction2 = vmap(self.norm_freq, interaction2)
            interaction2 = self.act_freq(interaction2)
            query_new = query_new + interaction2.mean(dim=1, keepdims=True)
            
            query_new = vmap(self.query_branch, query_new)
            context_new = vmap(self.context_branch, interaction+interaction2)
        else:
            query_new = vmap(self.query_branch, query_new)
            context_new = vmap(self.context_branch, interaction)
            
        return query_new, context_new


class SpatialFreqConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, padding_mode='reflect', norm_type=None, norm_kwargs=None, 
        act_type='RELU', act_kwargs=None, fft_type='ortho', reduction=2,):
        super().__init__()
        mid_channels = out_channels // reduction
        ffc_in_channels = ffc_out_channels = 2 * mid_channels
        conv_kwargs = dict(stride=1, padding_mode=padding_mode)
        block_kwargs = dict(norm_type=norm_type, norm_kwargs=norm_kwargs, act_type=act_type, act_kwargs=act_kwargs)
        self.fft_type = fft_type
        
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv_in = get_conv_block(
            in_channels, mid_channels, kernel_size=1, bias=False, adn_order='CNA', **conv_kwargs, **block_kwargs)
        self.conv = nn.Conv2d(ffc_in_channels, ffc_out_channels, kernel_size=1, **conv_kwargs)
        self.norm = get_norm_layer(norm_type=norm_type, args=[ffc_out_channels])
        self.act = get_act_layer(act_type=act_type, kwargs=act_kwargs)
        self.conv_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False, **conv_kwargs)
    
    def forward(self, x: torch.Tensor):
        out = self.spatial(x)
        x_fft = self.conv_in(x)
        ffted = torch.fft.rfftn(x_fft, dim=(-2, -1), norm=self.fft_type)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)  # (batch, c, h, w/2+1, 2)
        ffted = E.rearrange(ffted, "B C H W L -> B (C L) H W")
        
        ffted = self.conv(ffted)
        ffted = self.act(self.norm(ffted))
        ffted = E.rearrange(ffted, "B (C L) H W -> B C H W L", L=2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        
        ifft_shape_slice = x_fft.shape[2:]
        out_fft = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=(-2, -1), norm=self.fft_type)
        out_fft = self.conv_out(x_fft + out_fft)
        return out + out_fft
        

class RevisedLayerNorm(nn.Module):
    """
    Adapted from DehazeFormer, thanks!
    """
    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super().__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))
        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        nn.init.trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)
        nn.init.trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, x: torch.Tensor):
        is_context = x.ndim == 5
        if is_context:
            batch_size, support_size = x.shape[:2]
            x = E.rearrange(x, "B S ... -> (B S) ...")
        
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((x - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        normalized_x = (x - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_x * self.weight + self.bias
        
        if is_context:
            out = E.rearrange(out, "(B S) ... -> B S ...", B=batch_size, S=support_size)
            rescale = E.rearrange(rescale, "(B S) ... -> B S ...", B=batch_size, S=support_size)
            rebias = E.rearrange(rebias, "(B S) ... -> B S ...", B=batch_size, S=support_size)
            
        return out, rescale, rebias
    

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, use_rope=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_rope = use_rope
        
        if not use_rope:
            relative_positions = self.get_relative_positions(self.window_size)
            self.register_buffer("relative_positions", relative_positions)
            self.meta = nn.Sequential(
                nn.Linear(2, 256, bias=True),
                nn.GELU(),
                nn.Linear(256, num_heads, bias=True))
        else:
            self.relative_positions = RotaryEmbedding(head_dim, max_res=512)

    def forward(self, qkv:torch.Tensor, use_fast_attn=False):
        B_, N, _ = qkv.shape
        H = W = int(math.sqrt(N))
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv shape: [3, B_, Nh, N, C//N_h]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        if not self.use_rope:
            relative_position_bias = self.meta(self.relative_positions)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn_bias = relative_position_bias.unsqueeze(0)
        else:
            sin_emb, cos_emb = self.relative_positions.get_embed((H, W))
            q = apply_rot_embed(q, sin_emb, cos_emb)
            k = apply_rot_embed(k, sin_emb, cos_emb)
            attn_bias = None
        
        # q, k, v: [B_, Nh, N, C//N_h]
        if AVAIL_FAST_ATTN and use_fast_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale, attn_mask=attn_bias)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = (attn + attn_bias) if attn_bias is not None else attn
            attn = attn.softmax(dim=-1)
            x = attn @ v
            
        x = x.transpose(1, 2).reshape(B_, N, self.dim)
        
        return x

    def get_relative_positions(self, window_size):
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_positions_log  = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())
        return relative_positions_log


class ContextualMixer(nn.Module):
    """
    Artifact-aware contextual mixer
    """
    def __init__(
        self, network_depth, dim, num_heads, window_size, shift_size, 
        use_attn=False, use_rope=False, use_spectral=False, use_context=True,
        drop_proj=0., **block_kwargs):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size
        self.network_depth = network_depth  # for initialization of MLP
        self.use_attn = use_attn
        self.use_context = use_context
        block_kwargs_ = dict(norm_type=None, norm_kwargs=None, act_type='RELU', act_kwargs=None)
        block_kwargs_.update(block_kwargs)
        self.conv = CrossConvBlock((dim, dim), dim, use_spectral=use_spectral, **block_kwargs_) if use_context else SpatialFreqConv(dim, dim, padding_mode='reflect', **block_kwargs_)
        self.V = nn.Conv2d(dim, dim, 1)
        self.QK = nn.Conv2d(dim, dim * 2, 1) if use_attn else None
        self.attn = WindowAttention(dim, window_size, num_heads, use_rope=use_rope) if use_attn else None
        self.V_context = nn.Conv2d(dim, dim, 1) if use_context else None
        self.proj = nn.Conv2d(dim, dim, 1)
        self.drop_proj = nn.Dropout(drop_proj) if drop_proj > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape
            
            if w_shape[0] == self.dim * 2:    # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                nn.init.trunc_normal_(m.weight, std=std)        
            else:
                gain = (8 * self.network_depth) ** (-1/4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                nn.init.trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        
        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def window_partition(self, x, window_size):
        # "B H W C -> B H/win win W/win win C -> B H/win W/win win win C -> (B*H*W/win/win) win*win C"
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
    
    def forward(self, x, context=None, **attn_kwargs):
        # x: [B, Ciq, H, W], context: [B, S, Cic, H, W]
        H, W = x.shape[-2:]
        V = self.V(x)  # [B, Ciq, H, W]
        attn_out = None
        
        if self.use_attn:
            QK = self.QK(x)
            QKV = torch.cat([QK, V], dim=1)
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = self.window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C
            attn_windows = self.attn(qkv, **attn_kwargs)
            shifted_out = self.window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C
            attn_out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
            attn_out = attn_out.permute(0, 3, 1, 2)
        
        if self.use_context:
            assert context is not None
            V_context = vmap(self.V_context, context)  # [B, S, Cic, H, W]
            conv_out, context_out = self.conv(V.unsqueeze(dim=1), V_context)  # [B, S, Cic, H, W]
            conv_out = conv_out.squeeze(dim=1)  # [B, Cic, H, W]
            x_out = self.proj(conv_out) if attn_out is None else self.proj(conv_out + attn_out)
            x_out = self.drop_proj(x_out)
            return x_out, context_out
        else:
            conv_out = self.conv(V)
            x_out = self.proj(conv_out) if attn_out is None else self.proj(conv_out + attn_out)
            x_out = self.drop_proj(x_out)
            return x_out


class TransformerBlock(nn.Module):
    """
    norm -> attn -> res -> norm -> mlp -> res
    """
    def __init__(
        self, network_depth, dim, num_heads, mlp_ratio=4., norm_layer=RevisedLayerNorm, 
        cond_dim=24, window_size=8, shift_size=0, drop_proj=0., drop_path=0., 
        use_attn=True, use_rope=False, use_spectral=False, use_context=True,
        use_bounded_scale=False, use_scale_for_mlp=True, **block_kwargs):
        super().__init__()
        self.use_attn = use_attn
        self.use_context = use_context
        self.use_scale_for_mlp = use_scale_for_mlp
        self.norm = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = ContextualMixer(
            network_depth, dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size, drop_proj=drop_proj, 
            use_attn=use_attn, use_rope=use_rope, use_spectral=use_spectral, use_context=use_context, **block_kwargs)
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.local_scale = None
        if cond_dim > 0:
            self.local_scale = nn.Sequential(nn.Linear(cond_dim, dim, bias=True), nn.Tanh()) if use_bounded_scale else nn.Linear(cond_dim, dim, bias=True)
        
        self.norm_context = norm_layer(dim) if use_context and use_attn else nn.Identity()
        self.mlp_context = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio)) if use_context else nn.Identity()
        self.local_scale_context = None
        if use_context and cond_dim > 0:
            self.local_scale_context = nn.Sequential(nn.Linear(cond_dim, dim, bias=True), nn.Tanh()) if use_bounded_scale else nn.Linear(cond_dim, dim, bias=True)

        init_weights_with_scale([
            self.norm, self.attn, self.mlp, self.norm_context, self.mlp_context], 1)
        init_weights_with_scale([self.local_scale, self.local_scale_context], 0.1)
    
    
    def forward_no_context(self, x, cond=None, **attn_kwargs):
        if self.use_attn: 
            y, rescale, rebias = self.norm(x)
            y = self.attn(y, **attn_kwargs)
            y = y * rescale + rebias
        else:
            y = self.attn(y, **attn_kwargs)
        
        if cond is not None and self.local_scale is not None:
            local_scale = self.local_scale(cond)  # TODO: check size, check where to merge
            y = y * local_scale.view(-1, y.shape[1], 1, 1)
        x = x + self.drop_path(y)
        
        y = self.mlp(x)
        if cond is not None and self.local_scale is not None and self.use_scale_for_mlp:
            y = y * local_scale.view(-1, y.shape[1], 1, 1)
        x = x + self.drop_path(y)
        
        return x
    
    def forward(self, x, context=None, cond=None, **attn_kwargs):
        if context is None:
            return self.forward_no_context(x, cond=cond, **attn_kwargs)
        else:
            assert self.use_context
        
        if self.use_attn:
            y, rescale, rebias = self.norm(x)
            y_context, rescale_context, rebias_context = self.norm_context(context)
            y, y_context = self.attn(y, context=y_context, **attn_kwargs)
            y_context = y_context * rescale_context + rebias_context
            y = y * rescale + rebias
        else:
            y, y_context = self.attn(x, context=context, **attn_kwargs)
        
        if cond is not None and self.local_scale is not None:
            local_scale = self.local_scale(cond)  # TODO: check size, check where to merge
            local_scale_context = self.local_scale_context(cond)
            y = y * local_scale.view(-1, y.shape[1], 1, 1)
            y_context = y_context * local_scale_context.view(-1, 1, y.shape[1], 1, 1)
            
        x = x + self.drop_path(y)
        context = context + self.drop_path(y_context)

        y = self.mlp(x)
        y_context = vmap(self.mlp_context, context)
        
        if cond is not None and self.use_scale_for_mlp and self.local_scale is not None:
            y = y * local_scale.view(-1, y.shape[1], 1, 1)
            y_context = y_context * local_scale_context.view(-1, 1, y.shape[1], 1, 1)
        
        x = x + self.drop_path(y)
        context = context + self.drop_path(y_context)
            
        return x, context


class BasicLayer(nn.Module):
    def __init__(
        self, network_depth, dim, depth, num_heads, mlp_ratio=4., window_size=8, 
        attn_ratio=0., attn_loc='last', drop_proj=0., drop_path=0.,
        use_rope=False, use_spectral=True, cond_dim=24, use_context=True, 
        use_bounded_scale=False, use_scale_for_mlp=False, **block_kwargs):
        super().__init__()
        self.dim = dim
        self.depth = depth
        attn_depth = attn_ratio * depth
        norm_layer = RevisedLayerNorm
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        if attn_loc == 'last':
            use_attns = [i >= depth-attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        else:
            use_attns = [i >= (depth-attn_depth)//2 and i < (depth+attn_depth)//2 for i in range(depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                network_depth=network_depth,
                dim=dim, 
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                drop_proj=drop_proj,
                drop_path=dpr[i],
                use_attn=use_attns[i], 
                use_rope=use_rope,
                use_spectral=use_spectral,
                cond_dim=cond_dim, 
                use_context=use_context,
                use_bounded_scale=use_bounded_scale,
                use_scale_for_mlp=use_scale_for_mlp,
                **block_kwargs)
            for i in range(depth)])

    def forward(self, x, context=None, cond=None, **attn_kwargs):
        if context is not None:
            for blk in self.blocks:
                x, context = blk(x, context=context, cond=cond, **attn_kwargs)
            return x, context
        else:
            for blk in self.blocks:
                x = blk(x, cond=cond, **attn_kwargs)
            return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = patch_size if kernel_size is None else kernel_size
        padding = (kernel_size - patch_size + 1) // 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=kernel_size, stride=patch_size,
            padding=padding, padding_mode='reflect')

        init_weights_with_scale(self.proj, 0.1)
        
    def forward(self, x):
        return self.proj(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_channels=3, embed_dim=96, kernel_size=None, output_padding=0):
        super().__init__()
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        kernel_size = patch_size if kernel_size is None else kernel_size
        padding = (kernel_size - patch_size + output_padding) // 2
        self.proj = nn.ConvTranspose2d(
            embed_dim, out_channels, kernel_size=kernel_size, stride=patch_size,
            padding=padding, output_padding=output_padding)  
        # only zero-pad is supported for convt, so better not padding

        init_weights_with_scale(self.proj, 0.1)
        
    def forward(self, x):
        return self.proj(x)


class FeatureFuser(nn.Module):
    def __init__(self, in_dims, out_dim=None, fusion_type='cat', use_se_fusion=True, reduction=8):
        super().__init__()
        if fusion_type == 'add':
            assert all(d == in_dims[0] for d in in_dims)
            in_dim = in_dims[0]
        else:
            in_dim = sum(in_dims)
        out_dim = in_dim if out_dim is None else out_dim
        self.fusion_type = fusion_type
        if use_se_fusion:
            from modules.basic_blocks import SEBlock
            self.se = SEBlock(in_dim, reduction=reduction)
        else: 
            self.se = nn.Identity()
        self.conv = nn.Conv2d(in_dim, out_dim, 1, bias=False)
        init_weights_with_scale([self.se, self.conv])
    
    @staticmethod
    def assert_same_shape(tensor_tuple, dim='all'):
        shape = tensor_tuple[0].shape
        if dim == 'all':
            assert all(_.shape == shape for _ in tensor_tuple)
        else:
            assert all(_.shape[dim] == shape[dim] for _ in tensor_tuple)
    
    def forward(self, *in_feats):
        if self.fusion_type == 'cat':
            self.assert_same_shape(in_feats, dim=-3)
            x = torch.cat(in_feats, dim=-3)
        elif self.fusion_type == 'add':
            self.assert_same_shape(in_feats)
            for y in in_feats:
                x = torch.add(x, y)
        is_context = x.ndim == 5
        if is_context:
            batch_size, support_size = x.shape[:2]
            x = E.rearrange(x, "B S ... -> (B S) ...")
        x = self.se(x)
        x = self.conv(x)
        if is_context:
            x = E.rearrange(x, "(B S) ... -> B S ...", B=batch_size, S=support_size)
            
        return x


class Encoder(nn.Module):
    def __init__(
        self, in_channels=1,  window_size=8,
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[0, 1/2, 1, 0, 0],
        drop_proj_rates=[0., 0., 0., 0., 0.],
        drop_path_rates=[0., 0., 0., 0., 0.],
        use_rope=False,
        use_spectrals=[True, True, True, True, True],
        cond_dim=24, 
        use_bounded_scale=True, 
        use_scale_for_mlp=True,
        is_context_pair=False,
        **block_kwargs):
        super().__init__()
        
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios
        network_depth = sum(depths)
        common_params = dict(
            network_depth=network_depth, attn_loc='last', window_size=window_size, cond_dim=cond_dim, 
            use_bounded_scale=use_bounded_scale, use_scale_for_mlp=use_scale_for_mlp, use_rope=use_rope)
        layer_params = [
            dict(
                dim=embed_dims[i], depth=depths[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                attn_ratio=attn_ratio[i],  drop_proj=drop_proj_rates[i], drop_path=drop_path_rates[i], 
                use_spectral=use_spectrals[i]) for i in range(len(embed_dims))
        ]
        
        in_channels_context = in_channels if is_context_pair else 2*in_channels
        self.patch_embed_context = PatchEmbed(patch_size=1, in_channels=in_channels_context, embed_dim=embed_dims[0], kernel_size=3)
        self.enc_down1_context = PatchEmbed(patch_size=2, in_channels=embed_dims[0], embed_dim=embed_dims[1])  # downsampling
        self.enc_down2_context = PatchEmbed(patch_size=2, in_channels=embed_dims[1], embed_dim=embed_dims[2])  # downsampling
            
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(patch_size=1, in_channels=in_channels, embed_dim=embed_dims[0], kernel_size=3)
        
        self.enc_layer1 = BasicLayer(**common_params, **layer_params[0], **block_kwargs)
        self.enc_down1 = PatchEmbed(patch_size=2, in_channels=embed_dims[0], embed_dim=embed_dims[1])  # downsampling
        
        self.enc_layer2 = BasicLayer(**common_params, **layer_params[1], **block_kwargs)
        self.enc_down2 = PatchEmbed(patch_size=2, in_channels=embed_dims[1], embed_dim=embed_dims[2])  # downsampling
        
        # bottleneck
        self.neck = BasicLayer(**common_params, **layer_params[2], **block_kwargs)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x, context, cond=None, **attn_kwargs):
        y = self.patch_embed(x)  # [B, 24, H, W]
        c = vmap(self.patch_embed_context, context)
        
        y, c = self.enc_layer1(y, context=c, cond=cond, **attn_kwargs)  # [B, 24, H, W]
        s1 = y
        s1_context = c
        y = self.enc_down1(y)  # [B, 48, H/2, W/2]
        c = vmap(self.enc_down1_context, c)
        
        y, c = self.enc_layer2(y, cond=cond, context=c, **attn_kwargs)  # [B, 48, H/2, W/2]
        s2 = y
        s2_context = c
        y = self.enc_down2(y)  # [B, 96, H/4, W/4]
        c = vmap(self.enc_down2, c)
       
        y, c = self.neck(y, context=c, cond=cond, **attn_kwargs)  # [B, 96, H/4, W/4]
            
        return y, c, s1, s2, s1_context, s2_context

    
class Decoder(nn.Module):
    def __init__(
        self, out_channels=1, window_size=8,
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[0, 1/2, 1, 0, 0],
        drop_proj_rates=[0., 0., 0., 0., 0.],
        drop_path_rates=[0., 0., 0., 0., 0.],
        use_rope=False,
        use_spectrals=[True, True, True, True, True],
        fusion_type='cat',
        use_se_fusion=False,
        cond_dim=24,
        use_bounded_scale=True,
        use_scale_for_mlp=True,
        **block_kwargs):
        super().__init__()
        network_depth = sum(depths)
        assert fusion_type in ['cat', 'add'], f'Invalid fusion type: {fusion_type}.'
        fusion_dict = dict(fusion_type=fusion_type, use_se_fusion=use_se_fusion)
        common_params = dict(
            network_depth=network_depth, attn_loc='last', window_size=window_size, cond_dim=cond_dim, 
            use_bounded_scale=use_bounded_scale, use_scale_for_mlp=use_scale_for_mlp, use_rope=use_rope)
        layer_params = [
            dict(
                dim=embed_dims[i], depth=depths[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                attn_ratio=attn_ratio[i],  drop_proj=drop_proj_rates[i], drop_path=drop_path_rates[i], 
                use_spectral=use_spectrals[i]) for i in range(len(embed_dims))
        ]
        
        self.dec_up1_context = PatchUnEmbed(patch_size=2, out_channels=embed_dims[3], embed_dim=embed_dims[2])
        self.dec_up2_context = PatchUnEmbed(patch_size=2, out_channels=embed_dims[4], embed_dim=embed_dims[3])
        self.dec_fuse1_context = FeatureFuser((embed_dims[3], embed_dims[3]), embed_dims[3], **fusion_dict)
        self.dec_fuse2_context = FeatureFuser((embed_dims[4], embed_dims[4]), embed_dims[4], **fusion_dict)
        
        self.dec_up1 = PatchUnEmbed(patch_size=2, out_channels=embed_dims[3], embed_dim=embed_dims[2])  # upsampling
        self.dec_fuse1 = FeatureFuser((embed_dims[3], embed_dims[3]), embed_dims[3], **fusion_dict)
        self.dec_layer1 = BasicLayer(**common_params, **layer_params[3], **block_kwargs)
        
        self.dec_up2 = PatchUnEmbed(patch_size=2, out_channels=embed_dims[4], embed_dim=embed_dims[3])  # upsampling
        self.dec_fuse2 = FeatureFuser((embed_dims[4], embed_dims[4]), embed_dims[4], **fusion_dict)
        self.dec_layer2 = BasicLayer(**common_params, **layer_params[4], **block_kwargs)

        self.patch_unembed = PatchUnEmbed(patch_size=1, out_channels=out_channels, embed_dim=embed_dims[4], kernel_size=3)
    
    def forward(self, y, context, s1, s2, s1_context, s2_context, cond=None, **attn_kwargs):
        x = self.dec_up1(y)  # [B, 48, H/2, W/2]
        context = vmap(self.dec_up1_context, context)
        
        x = self.dec_fuse1(x, s2) + x  # [B, 48, H/2, W/2]
        context = self.dec_fuse1_context(context, s2_context) + context
        
        x, context = self.dec_layer1(x, context=context, cond=cond, **attn_kwargs)  # [B, 48, H/2, W/2]
        
        x = self.dec_up2(x)  # [B, 24, H, W]
        context = vmap(self.dec_up2_context, context)
        
        x = self.dec_fuse2(x, s1) + x  # [B, 24, H, W]
        context = self.dec_fuse2_context(context, s1_context) + context
        
        x, _ = self.dec_layer2(x, context=context, cond=cond, **attn_kwargs)  # [B, 24, H, W]
        
        x = self.patch_unembed(x)  # [B, C_in, H, W]
        return x
    

class ProCT(nn.Module):
    """
    This version differs from `sources/proct.py` in that it does not inherit
    methods from `BasicWrapper` class which is built upon torch-radon.
    Apart from that, everything is identical with `sources/proct.py`.
    """
    def __init__(
        self, in_channels=1, out_channels=1, 
        window_size=8, 
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[0, 1/2, 1, 0, 0],
        drop_proj_rates=[0., 0., 0., 0., 0.],
        drop_path_rates=[0., 0., 0., 0., 0.],
        cond_dim=24,
        use_rope=False,
        use_spectrals=[True, True, True, False, False],
        fusion_type='cat', 
        use_se_fusion=False,
        use_learnable_prompt=False, 
        use_bounded_scale=True,
        use_scale_for_mlp=True,
        use_global_scale=False,
        is_context_pair=False,
        block_kwargs={},
        num_full_views=720):  # This is a newly-added argument
        super().__init__()
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios
        self.cond_dim = cond_dim
        self.num_full_views = num_full_views
        
        assert embed_dims[1] == embed_dims[3]
        assert embed_dims[0] == embed_dims[4]
        use_spectrals = ensure_tuple_rep(use_spectrals, 5)
        drop_path_rates = ensure_tuple_rep(drop_path_rates, 5)
        drop_proj_rates = ensure_tuple_rep(drop_proj_rates, 5)
        former_dict = dict(
            window_size=window_size, embed_dims=embed_dims, mlp_ratios=mlp_ratios, depths=depths, num_heads=num_heads, 
            attn_ratio=attn_ratio, drop_proj_rates=drop_proj_rates, drop_path_rates=drop_path_rates, use_rope=use_rope,
            use_spectrals=use_spectrals, use_bounded_scale=use_bounded_scale, use_scale_for_mlp=use_scale_for_mlp,
            cond_dim=cond_dim)
        encoder_dict = dict(is_context_pair=is_context_pair)
        decoder_dict = dict(fusion_type=fusion_type, use_se_fusion=use_se_fusion)
        self.encoder = Encoder(in_channels=in_channels, **encoder_dict, **former_dict, **block_kwargs)
        self.decoder = Decoder(out_channels=out_channels, **decoder_dict, **former_dict, **block_kwargs)
        
        if cond_dim > 0:
            self.squeeze = nn.Linear(self.num_full_views, cond_dim, bias=True)
            self.global_scale = nn.Linear(cond_dim, out_channels, bias=True) if use_global_scale else None
            init_weights_with_scale([self.squeeze, self.global_scale], 0.1)
        else:
            self.squeeze = self.global_scale = None
        
        if use_learnable_prompt:
            # Dimension [1, 2, 5, num_full_views]
            # - 2 for two tasks: sparse-view and limited-angle, this can be modified.
            # - 5 for five levels: from easy to hard.
            self.prompt = nn.Parameter(torch.zeros(1, 2, 5, self.num_full_views).float(), requires_grad=True)
        else:
            self.prompt = None

    @staticmethod
    def pad_image(x, patch_size):
        h, w = x.shape[-2:]
        mod_pad_h = (patch_size - h % patch_size) % patch_size
        mod_pad_w = (patch_size - w % patch_size) % patch_size
        
        if mod_pad_h > 0 or mod_pad_w > 0:
            is_context = x.ndim == 5
            if is_context:
                batch_size, support_size = x.shape[:2]
                x = E.rearrange(x, "B S ... -> (B S) ...")
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            if is_context:
                x = E.rearrange(x, "(B S) ... -> B S ...", B=batch_size, S=support_size)
        
        return x
    
    def forward(self, x, context, cond=None, task_ids=None, global_skip=False, **attn_kwargs):
        # - x: the image to be reconstructed. [B, 1, H, W]
        # - context: the contextual pair(s), can be a simple phantom. [B, S, 2, H, W]
        # - cond: the view-aware prompts. [B, 1, Nv]
        # - task_ids: a list used to select the prompt components, with 
        #   task_ids[0] specifying the main task and task_ids[1] the sub-task.
        #   This is of no use if not using learnable prompt.
        if self.squeeze is not None:
            if self.prompt is not None and task_ids is not None:
                task_ids = torch.tensor(task_ids).to(cond.device)
                prompt = self.prompt.repeat(x.shape[0], 1, 1, 1)  # [B, Nt, Nts, Nv]
                prompt = prompt.index_select(dim=1, index=task_ids[0])  # [B, 1, Nts, Nv]
                prompt = prompt.index_select(dim=2, index=task_ids[1])   # [B, 1, 1, Nv]
                prompt = prompt.squeeze(dim=1).squeeze(dim=1)  # [B, Nv]
                cond = torch.sigmoid(prompt)
            cond = self.squeeze(cond).reshape(cond.shape[0], -1).contiguous()
            
        H, W = x.shape[2:]
        x = self.pad_image(x, self.patch_size)
        context = self.pad_image(context, self.patch_size)
        y, context_new, s1, s2, s1_context, s2_context = self.encoder(x, context=context, cond=cond, **attn_kwargs)
        y = self.decoder(y, context_new, s1, s2, s1_context, s2_context, cond=cond, **attn_kwargs)
        dim = y.shape[1]
        
        if cond is not None and self.global_scale is not None:
            global_scale = self.global_scale(cond)
            y = y * global_scale.view(-1, dim, 1, 1)
        
        if global_skip:
            y = y + x
        
        return y[:, :, :H, :W]


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    batch_size = 1
    num_support = 1
    x = torch.randn((batch_size, 1, 256, 256))#.cuda()
    context = torch.randn((batch_size, num_support, 2, 256, 256))#.cuda()
    prompt = torch.randn((batch_size, 1, 720))#.cuda()
    net_dict = {
        'use_spectrals':[True,True,True,False,False], 
        'use_bounded_scale':True, 
        'use_prompt':False, 
        'embed_dims':[24,48,96,48,24],
        'mlp_ratios':[2,4,4,2,2],
        'depths':[8,8,8,4,4],
        'num_heads':[2,4,6,1,1],
        'attn_ratio':[0,1/2,1,0,0],
        'fusion_type':'cat',
        'block_kwargs': {'norm_type':'INSTANCE'},
        }
    wrapper_kwargs = {'img_size':256, 'simul_poisson_rate':1e6, 'simul_gaussian_rate':0.01}
    net = ProCT(**net_dict, **wrapper_kwargs)#.cuda()
    y = net(x, context, prompt, [1])

    
 