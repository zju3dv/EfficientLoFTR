"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import torch
from torch.nn import Module
import torch.nn.functional as F
from einops.einops import rearrange

if hasattr(F, 'scaled_dot_product_attention'):
    FLASH_AVAILABLE = True
    from torch.backends.cuda import sdp_kernel
else:
    FLASH_AVAILABLE = False

def crop_feature(query, key, value, x_mask, source_mask):
    mask_h0, mask_w0, mask_h1, mask_w1 = x_mask[0].sum(-2)[0], x_mask[0].sum(-1)[0], source_mask[0].sum(-2)[0], source_mask[0].sum(-1)[0]
    query = query[:, :mask_h0, :mask_w0, :]
    key = key[:, :mask_h1, :mask_w1, :]
    value = value[:, :mask_h1, :mask_w1, :]
    return query, key, value, mask_h0, mask_w0

def pad_feature(m, mask_h0, mask_w0, x_mask):
    bs, L, H, D = m.size()
    m = m.view(bs, mask_h0, mask_w0, H, D)
    if mask_h0 != x_mask.size(-2):
        m = torch.cat([m, torch.zeros(m.size(0), x_mask.size(-2)-mask_h0, x_mask.size(-1), H, D, device=m.device, dtype=m.dtype)], dim=1)
    elif mask_w0 != x_mask.size(-1):
        m = torch.cat([m, torch.zeros(m.size(0), x_mask.size(-2), x_mask.size(-1)-mask_w0, H, D, device=m.device, dtype=m.dtype)], dim=2)
    return m

class Attention(Module):
    def __init__(self, no_flash=False, nhead=8, dim=256, fp32=False):
        super().__init__()
        self.flash = FLASH_AVAILABLE and not no_flash
        self.nhead = nhead
        self.dim = dim
        self.fp32 = fp32
        
    def attention(self, query, key, value, q_mask=None, kv_mask=None):
        assert q_mask is None and kv_mask is None, "Not support generalized attention mask yet."
        if self.flash and not self.fp32:
            args = [x.contiguous() for x in [query, key, value]]
            with sdp_kernel(enable_math= False, enable_flash= True, enable_mem_efficient= False):
                out = F.scaled_dot_product_attention(*args)
        elif self.flash:
            args = [x.contiguous() for x in [query, key, value]]
            out = F.scaled_dot_product_attention(*args)
        else:
            QK = torch.einsum("nlhd,nshd->nlsh", query, key)
    
            # Compute the attention and the weighted average
            softmax_temp = 1. / query.size(3)**.5  # sqrt(D)
            A = torch.softmax(softmax_temp * QK, dim=2)

            out = torch.einsum("nlsh,nshd->nlhd", A, value)
        return out

    def _forward(self, query, key, value, q_mask=None, kv_mask=None):
        if q_mask is not None:
            query, key, value, mask_h0, mask_w0 = crop_feature(query, key, value, q_mask, kv_mask)

        if self.flash:
            query, key, value = map(lambda x: rearrange(x, 'n h w (nhead d) -> n nhead (h w) d', nhead=self.nhead, d=self.dim), [query, key, value])
        else:
            query, key, value = map(lambda x: rearrange(x, 'n h w (nhead d) -> n (h w) nhead d', nhead=self.nhead, d=self.dim), [query, key, value])

        m = self.attention(query, key, value, q_mask=None, kv_mask=None)

        if self.flash:
            m = rearrange(m, 'n nhead L d -> n L nhead d', nhead=self.nhead, d=self.dim)

        if q_mask is not None:
            m = pad_feature(m, mask_h0, mask_w0, q_mask)
        
        return m
    
    def forward(self, query, key, value, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            if FLASH_AVAILABLE: # pytorch scaled_dot_product_attention
                queries: [N, H, L, D]
                keys: [N, H, S, D]
                values: [N, H, S, D]
            else:
                queries: [N, L, H, D]
                keys: [N, S, H, D]
                values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        bs = query.size(0)
        if bs == 1 or q_mask is None:            
            m = self._forward(query, key, value, q_mask=q_mask, kv_mask=kv_mask)
        else: # for faster trainning with padding mask while batch size > 1
            m_list = []
            for i in range(bs):
                m_list.append(self._forward(query[i:i+1], key[i:i+1], value[i:i+1], q_mask=q_mask[i:i+1], kv_mask=kv_mask[i:i+1]))
            m = torch.cat(m_list, dim=0)
        return m