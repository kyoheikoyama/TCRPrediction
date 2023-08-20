import sys
sys.path.append('../')
from recipes.model import *
from scipy.special import expit, softmax
from sklearn.metrics import roc_auc_score
import os
import numpy as np

# fix random seeds
seed = 9
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def get_attention_weights(__xx, model, explain_model, device='cpu'):
    #print(__xx)
    explain_model.load_state_dict(model.state_dict(), strict=False)
    model.to(device)
    explain_model = explain_model.to(device)
    model = model.eval()
    explain_model = explain_model.eval()
    
    with torch.no_grad():
        tcrab, epitope, src_kwargs, tgt_kwargs = explain_model(__xx)  #  (L,B,E)
        attn_output1, attn_output_weights1 = model.enc.cgr.mha.fg.f.mha(
            query=tcrab,  # target  27
            key=epitope,   # source  10  (source is usually the output of encoder)
            value=epitope,   # source  10  
            **{'key_padding_mask': tgt_kwargs['memory_key_padding_mask']}
        )

        attn_output2, attn_output_weights2 = model.enc.cgr.mha.fg.g.f.mha(
            query=epitope, 
            key=tcrab, 
            value=tcrab,
            **{'key_padding_mask': src_kwargs['memory_key_padding_mask']}
        )
        
        ypred = model(__xx)
    
    return (attn_output_weights1[0].cpu().numpy(), 
                             attn_output_weights2[0].cpu().numpy(),  
                            float(softmax(ypred.cpu().numpy())[0][1]))
    
    #return attn_output_weights1, attn_output_weights2, ypred


class Explain_TCRModel(nn.Module):
    def __init__(self, d_model, d_ff, n_head, n_local_encoder, n_global_encoder, dropout, scope, n_tok, n_pos1, n_pos2, n_seg, mode='attn'):
        super().__init__()
#         n_tok = 24  # NUM_VOCAB
#         n_pos1 = 46  # MAX_LEN_AB
#         n_pos2 = 21  # MAX_LEN_Epitope
        padding_idx = 0

        self.padding_idx = padding_idx
        self.d_model = d_model
        self.emb = TwinEmb(d_model, n_tok, n_pos1, n_pos2, n_seg, dropout)
        self.enc = Explain_Before_Cross(d_model, n_head, n_global_encoder, d_ff, dropout, scope,
                           n_local_encoder, mode, n_pos1, n_pos2)
        # self.exe = nn.Linear(2 * self.d_model, 2, bias=True)
        
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
        h_src, h_tgt = self.enc(x_src, x_tgt, src_kwargs, tgt_kwargs)   
        return h_src, h_tgt, src_kwargs, tgt_kwargs
    
class Explain_Before_Cross(nn.Module):
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
        return src, tgt

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
class Explain_SelfOnAll(nn.Module):
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
        h_concat = self.atten(h_concat, h_concat)
        h_all = h_concat.mean(dim=1)
        y = self.exe(h_all)
        return y