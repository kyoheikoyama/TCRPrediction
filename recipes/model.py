import copy

'''Blocks'''

import torch
import torch.nn as nn
import sys
sys.path.append('../')
from recipes.modelutils import ModifiedMultiheadAttention


class PFF(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.lin1 = nn.Linear(d_model, dim_feedforward)
        self.lin2 = nn.Linear(dim_feedforward, d_model)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        return self.lin2(self.drop(self.relu(self.lin1(x))))


class MHA(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.mha = ModifiedMultiheadAttention(d_model, nhead, dropout)
        self.att = None

    def forward(self, tgt, src, **kwargs):
        # - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
        #   the embedding dimension.
        # - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
        #   the embedding dimension.
        # - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
        #   the embedding dimension.
        # - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.

        tgt, att = self.mha(query=tgt, key=src, value=src, **kwargs)  # q, k, v
        self.att = att
        return tgt


class CNV(nn.Module):
    """
    L_o = L_i + 2 * padding - dilation * (kernel_size -1)
    L_o = L_i => padding = dilation * (kernel_size -1) / 2
    """
    def __init__(self, d_model, dropout, scope, scale):
        super().__init__()
        kernel_size = 2 * scope * scale - 1  # odd
        dilation = 1
        padding = (dilation * (kernel_size - 1)) // 2
        self.conv = nn.Conv1d(d_model, d_model,
                              kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)  # (L,B,E) --> (B,L,E)
        x = self.norm(x)
        x = torch.transpose(x, 0, 1)  # (B,L,E) --> (B,E,L)
        x = x.permute(1,2,0)  # (L,B,E) --> (B,E,L)
        x = self.conv(x)
        x = x.permute(2,0,1)  #  (B,E,L) --> (L,B,E)
        x = self.relu(x)
        x = self.drop(x)
        return x


class TransformerEncorder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                   dropout, activation='gelu')
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers,
                                             encoder_norm)
        # Remark: TransformerEncoder is a wrapper

    def forward(self, x, mask, src_key_padding_mask):
        x = self.encoder(x, mask, src_key_padding_mask)
        return x


'''Blocks'''
class Swap(nn.Module):
    """ Swap: Y2Y
    Cf. Twist: II2X
    """
    def __init__(self, f):
        super().__init__()
        self.add_module('f', f)

    def forward(self, u, v, **kwargs):
        return self.f(v, u, **kwargs)


class Parallel(nn.Module):
    """ Parallel: II2II
    f: U --> U
    g: V --> V
    """
    def __init__(self, f, g):
        super().__init__()
        self.add_module('f', f)
        self.add_module('g', g)

    def forward(self, u, v, f_kwargs, g_kwargs):
        return self.f(u, **f_kwargs), self.g(v, **g_kwargs)


class Entangle(nn.Module):
    """ Entangle: YY2X
    f: U x V --> U
    g: U x V --> V
    """
    def __init__(self, f, g):
        super().__init__()
        self.add_module('f', f)
        self.add_module('g', g)

    def forward(self, u, v, f_kwargs, g_kwargs):
        return self.f(u, v, **f_kwargs), self.g(u, v, **g_kwargs)


class Residual(nn.Module):
    """ X2X (incl. (II)2(II))
        I2I: TODO (Integrate _Residual)
    """
    def __init__(self, fg, d):
        super().__init__()
        self.add_module('fg', fg)
        self.norm_u = nn.LayerNorm(d)
        self.norm_v = nn.LayerNorm(d)

    def forward(self, u, v, f_kwargs, g_kwargs):
        _u, _v = u, v
        u, v = self.fg(self.norm_u(u), self.norm_u(v), f_kwargs, g_kwargs)
        return u + _u, v + _v


class _Residual(nn.Module):
    def __init__(self, fg, d):
        super().__init__()
        self.add_module('fg', fg)
        self.norm_u = nn.LayerNorm(d)

    def forward(self, u, **f_kwargs):
        _u = u
        u = self.fg(self.norm_u(u), **f_kwargs)
        return u + _u



class SelfLocalReferenceConv(nn.Module):
    def __init__(self, d_model, d_length, scope, n_local_encoder, dropout):
        super().__init__()
        self.f = nn.Sequential(
            CNV(d_model, dropout, scope, scale=1),
            CNV(d_model, dropout, scope, scale=4),
            CNV(d_model, dropout, scope, scale=8)
        )
        
    def forward(self, x, **kwargs):
        x = self.f(x)
        return x

    
class SelfLocalReferenceAttn(nn.Module):
    """ https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py#L164
        https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3912
        
        if attn_mask is not None:
            attn_output_weights += attn_mask
            
        error: in binary_cross_entropy
        RuntimeError: reduce failed to synchronize: cudaErrorAssert: device-side assert triggered
        --> almost zero...?
        
        e.g. torch.ones(2,3,4,4) + generate_band_matrix(torch.ones(4, 3, 2), 2)
    """
    def __init__(self, d_model, d_length, scope, n_local_encoder, dropout):
        super().__init__()
        # scales = [2 ** i for i in range(n_local_encoder)]
        self.rs = [20 for _ in range(n_local_encoder)]
        self.fs = nn.ModuleList([copy.deepcopy(
            TransformerEncorder(d_model, nhead=8, num_encoder_layers=1, dim_feedforward=d_model//2, dropout=dropout)
        ) for _ in range(n_local_encoder)])
        
    def forward(self, x, **kwargs):
        for f, r in zip(self.fs, self.rs):
            mask = self.generate_band_mask(x, r)
            x = f(x, mask, **kwargs)
        return x
    
    def generate_band_matrix(self, x, r):
        d = x.size(0)
        m = min(d, r)
        s = torch.ones_like(x[:, 0, 0])  # don't use the torch.eye(x.size(0)) - multi gpu error
        e = torch.diag(s)
        u = sum([torch.diag(s[:d-i],diagonal=i) for i in range(1, m)])
        l = u.transpose(0,1)
        return e + u + l
    
    def generate_band_mask(self, x, r):
        r"""The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = self.generate_band_matrix(x, r)
        mask = mask.float(
        ).masked_fill(
            mask == 0, -1. * float(2**32)  # can't we use float('-inf') with pytorch and gpus?
        ).masked_fill(
            mask == 1, float(0.0))
        return mask

    
class SelfLocalReferenceIdnt(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Identity()
        
    def forward(self, x, **kwargs):
        x = self.f(x)
        return x
    

class SelfLocalReference(nn.Module):
    """ Switch pattern (to be a wrapper) """
    def __init__(self, d_model, d_length, scope, n_local_encoder, dropout, mode='attn'):
        super().__init__()
        if mode == 'conv':
            self.f = SelfLocalReferenceConv(d_model, d_length, scope, n_local_encoder, dropout)
        elif mode == 'attn':
            self.f = SelfLocalReferenceAttn(d_model, d_length, scope, n_local_encoder, dropout)
        else:
            self.f = SelfLocalReferenceIdnt()
        
    def forward(self, x, **kwargs):
        x = self.f(x, **kwargs)
        return x


class SelfGlobalReference(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.f = TransformerEncorder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        
    def forward(self, x, **kwargs):
        x = self.f(x, **kwargs)
        return x

    
class ParallelCrossGlobalReference(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__() 
        self.mha = Residual(Entangle(
            MHA(d_model, nhead, dropout), Swap(MHA(d_model, nhead, dropout))
        ), d_model)
        self.pff = Residual(Parallel(
            PFF(d_model, dim_feedforward, dropout),
            PFF(d_model, dim_feedforward, dropout)
        ), d_model)

    def forward(self, src, tgt, src_kwargs, tgt_kwargs):
        src, tgt = self.mha(src, tgt, src_kwargs, tgt_kwargs)
        src, tgt = self.pff(src, tgt, {}, {})
        return src, tgt


class TwinEmb(nn.Module):
    def __init__(self, d_model, n_tok, n_pos1, n_pos2, n_seg, dropout, padding_idx=0):
        super().__init__()
        self.f = Parallel(
            EMB(d_model, n_tok, n_pos1, n_seg, dropout, padding_idx),
            EMB(d_model, n_tok, n_pos2, n_seg, dropout, padding_idx),
        )
        
    def forward(self, u, v, f_kwargs, g_kwargs):
        return self.f(u, v, f_kwargs, g_kwargs)

    
class TwinEnc(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout,
                 scope, n_local_encoder, mode, n_pos1, n_pos2):
        super().__init__()
        self.slr = Parallel(
            SelfLocalReference(d_model, n_pos1-1, scope, n_local_encoder, dropout, mode),
            SelfLocalReference(d_model, n_pos2-1, scope, n_local_encoder, dropout, mode)
        )
        self.sgr = Parallel(
            SelfGlobalReference(d_model, nhead,
                                num_encoder_layers, dim_feedforward, dropout),
            SelfGlobalReference(d_model, nhead,
                                num_encoder_layers, dim_feedforward, dropout)
        )
        self.cgr = ParallelCrossGlobalReference(d_model, nhead,
                                                dim_feedforward, dropout)
        self._reset_parameters()
        
    def forward(self, src, tgt, src_kwargs, tgt_kwargs):
        kwargs_slr_src = {
            'src_key_padding_mask': src_kwargs['src_key_padding_mask']
        }
        kwargs_slr_tgt = {
            'src_key_padding_mask': src_kwargs['tgt_key_padding_mask']
        }
        kwargs_sgr_src = {
            'mask': None,
            'src_key_padding_mask': src_kwargs['src_key_padding_mask']
        }
        kwargs_sgr_tgt = {
            'mask': None,
            'src_key_padding_mask': src_kwargs['tgt_key_padding_mask']
        }
        kwargs_cgr = {
            'attn_mask': None,
            'key_padding_mask': None  # TODO: src_kwargs['src_key_padding_mask']}
        }

        src, tgt = src.permute(1,0,2), tgt.permute(1,0,2)  # (B,L,E) --> (L,B,E)
        src, tgt = self.slr(src, tgt, kwargs_slr_src, kwargs_slr_tgt)
        # src, tgt = self.sgr(src, tgt, kwargs_sgr_src, kwargs_sgr_tgt)
        src, tgt = self.cgr(src, tgt, kwargs_cgr, kwargs_cgr)
        src, tgt = src.permute(1,0,2), tgt.permute(1,0,2)  # (L,B,E) --> (B,L,E)

        return src, tgt

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class EMB(nn.Module):
    def __init__(self, d_model, n_tok, n_pos, n_seg, dropout, padding_idx=0):
        super().__init__()
        import math
        self.scale = math.sqrt(d_model)
        self.d_model = d_model
        self.tok_emb = nn.Embedding(n_tok, d_model, padding_idx)
        self.pos_emb = nn.Embedding(n_pos, d_model, padding_idx)
        self.seg_emb = nn.Embedding(n_seg, d_model, padding_idx)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, x, **kwargs):
        t, p, s = x  # (B,L)
        t = self.tok_emb(t)
        p = self.pos_emb(p)
        s = self.seg_emb(s)
        x = (t + p + s) * self.scale
        x = self.drop(x)
        return x  # (B,L,E)

    def init_weights(self):
        initrange = 0.1
        self.tok_emb.weight.data.uniform_(-initrange, initrange)
        self.pos_emb.weight.data.uniform_(-initrange, initrange)
        self.seg_emb.weight.data.uniform_(-initrange, initrange)

class SelfOnAll(nn.Module):
    def __init__(self, d_model, d_ff, n_head, n_local_encoder, n_global_encoder, dropout, scope, mode='attn', 
                n_tok=24, n_pos1=50, n_pos2=25, n_seg=5):
        super().__init__()
        padding_idx = 0
        self.padding_idx = padding_idx
        self.d_model = d_model
        self.emb = TwinEmb(d_model, n_tok, n_pos1, n_pos2, n_seg, dropout)
        self.enc = TwinEnc(d_model, n_head, n_global_encoder, d_ff, dropout, scope,
                           n_local_encoder, mode, n_pos1, n_pos2)
        self.atten = MHA(d_model, n_head, dropout)
        self.exe = nn.Linear(self.d_model, 2, bias=True)
        
    def forward(self, src_tgt):
        src, tgt = src_tgt
        pad_mask_src = src[:,0,:] == self.padding_idx
        pad_mask_tgt = tgt[:,0,:] == self.padding_idx
        src_kwargs = {  # for src_tgt
            'src_mask': None,  # for self attention (src)
            'tgt_mask': None,  # for self attention (tgt)
            'memory_mask': None,  # for source-target attention (tgt)
            'src_key_padding_mask': pad_mask_src,  # for self attention (src)
            'tgt_key_padding_mask': pad_mask_tgt,  # for self attention (tgt)
            'memory_key_padding_mask': pad_mask_src  # for source-target attention (tgt)
        }
        tgt_kwargs = {  # for tgt_src
            'src_mask': None,  # for self attention (src)
            'tgt_mask': None,  # for self attention (tgt)
            'memory_mask': None,  # for source-target attention (tgt)
            'src_key_padding_mask': pad_mask_tgt,  # for self attention (src)
            'tgt_key_padding_mask': pad_mask_src,  # for self attention (tgt)
            'memory_key_padding_mask': pad_mask_tgt  # for source-target attention (tgt)
        }
        
        s_src, s_tgt = (src[:,0,:], src[:,1,:], src[:,2,:]), (tgt[:,0,:], tgt[:,1,:], tgt[:,2,:])
        x_src, x_tgt = self.emb(s_src, s_tgt, {}, {})
        h_src, h_tgt = self.enc(x_src, x_tgt, src_kwargs, tgt_kwargs)   # (B,L,E)
        h_concat = torch.cat([h_src, h_tgt], dim=1)
        h_concat = h_concat.permute(1,0,2)  # (B,L,E) --> (L,B,E)

        keypaddingmask = torch.cat([pad_mask_src, pad_mask_tgt], dim=1)

        h_concat = self.atten(h_concat, h_concat, 
                              key_padding_mask=keypaddingmask
                              )
        h_concat = h_concat.permute(1,0,2)  # (L,B,E) --> (B,L,E)
        
        h_all = h_concat.mean(dim=1)
        # print("h_all.shape", h_all.shape)
        y = self.exe(h_all)
        return y

class TCRModel(nn.Module):
    def __init__(self, d_model, d_ff, n_head, n_local_encoder, n_global_encoder, dropout, scope, mode='attn', 
                n_tok=24, n_pos1=50, n_pos2=25, n_seg=5):
        super().__init__()
        padding_idx = 0
        self.padding_idx = padding_idx
        self.d_model = d_model
        self.emb = TwinEmb(d_model, n_tok, n_pos1, n_pos2, n_seg, dropout)
        self.enc = TwinEnc(d_model, n_head, n_global_encoder, d_ff, dropout, scope,
                           n_local_encoder, mode, n_pos1, n_pos2)
        self.exe = nn.Linear(2 * self.d_model, 2, bias=True)
        
    def forward(self, src_tgt):
        src, tgt = src_tgt
        pad_mask_src = src[:,0,:] == self.padding_idx
        pad_mask_tgt = tgt[:,0,:] == self.padding_idx
        
        src_kwargs = {  # for src_tgt
            'src_mask': None,  # for self attention (src)
            'tgt_mask': None,  # for self attention (tgt)
            'memory_mask': None,  # for source-target attention (tgt)
            'src_key_padding_mask': pad_mask_src,  # for self attention (src)
            'tgt_key_padding_mask': pad_mask_tgt,  # for self attention (tgt)
            'memory_key_padding_mask': pad_mask_src  # for source-target attention (tgt)
        }
        tgt_kwargs = {  # for tgt_src
            'src_mask': None,  # for self attention (src)
            'tgt_mask': None,  # for self attention (tgt)
            'memory_mask': None,  # for source-target attention (tgt)
            'src_key_padding_mask': pad_mask_tgt,  # for self attention (src)
            'tgt_key_padding_mask': pad_mask_src,  # for self attention (tgt)
            'memory_key_padding_mask': pad_mask_tgt  # for source-target attention (tgt)
        }
        
        s_src, s_tgt = (src[:,0,:], src[:,1,:], src[:,2,:]), (tgt[:,0,:], tgt[:,1,:], tgt[:,2,:])
        x_src, x_tgt = self.emb(s_src, s_tgt, {}, {})
        h_src, h_tgt = self.enc(x_src, x_tgt, src_kwargs, tgt_kwargs)   # (B,L,E)
        h_src, h_tgt = h_src.mean(dim=1), h_tgt.mean(dim=1)
        y = self.exe(torch.cat([h_src, h_tgt], dim=1))
        # y_src, y_tgt = self.exe(h_src, h_tgt, {}, {})
        return y
    
class SPBModel(nn.Module):
    def __init__(self, d_model, d_ff, n_head, n_local_encoder, n_global_encoder, dropout, scope, mode='attn', 
                n_tok=24, n_pos1=50, n_pos2=25, n_seg=5):
        super().__init__()
        padding_idx = 0
        self.padding_idx = padding_idx
        self.d_model = d_model
        self.emb = TwinEmb(d_model, n_tok, n_pos1, n_pos2, n_seg, dropout)
        self.enc = TwinEnc(d_model, n_head, n_global_encoder, d_ff, dropout, scope,
                           n_local_encoder, mode, n_pos1, n_pos2)
        self.exe = nn.Linear(self.d_model, 2, bias=True)
        
    def forward(self, src_tgt):
        src, _ = src_tgt
        pad_mask_src = src[:,0,:] == self.padding_idx
        
        src_kwargs = {  # for src_tgt
            'src_mask': None,  # for self attention (src)
            'tgt_mask': None,  # for self attention (tgt)
            'memory_mask': None,  # for source-target attention (tgt)
            'src_key_padding_mask': pad_mask_src,  # for self attention (src)
            'tgt_key_padding_mask': pad_mask_src,  # for self attention (tgt)
            'memory_key_padding_mask': pad_mask_src  # for source-target attention (tgt)
        }
        
        s_src = (src[:,0,:], src[:,1,:], src[:,2,:])
        x_src, _ = self.emb(s_src, s_src, {}, {})
        h_src, h_src = self.enc(x_src, x_src, src_kwargs, src_kwargs)   # (B,L,E)
        h_src, h_src = h_src.mean(dim=1), h_src.mean(dim=1)
        y = self.exe(torch.cat([h_src], dim=1))
        # y_src, y_tgt = self.exe(h_src, h_tgt, {}, {})
        return y

