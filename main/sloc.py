
## input_ids tensor([[  101,  1045,  2001,  3110,  2428, 11587,  1998,  2091,  2058,  2054, 2026,  3611,  2056,   102]], device='cuda:0')
#logits tensor([[ 6.7980, -1.6343, -1.4902, -0.8498, -0.9241, -1.8199]], device='cuda:0')
#attr-scores: tensor([ 0.0000, -0.0239,  0.0182,  0.3137,  0.0000,  0.7476,  0.2085,  0.0019, 0.0000, -0.0074, -0.0612, -0.2065,  0.0054,  0.1682], device='cuda:0')
#attr-scores-shape: torch.Size([14])
#logits: tensor([[ 6.7980, -1.6343, -1.4902, -0.8498, -0.9241, -1.8199]], device='cuda:0')

import torch
import sys
from scipy.sparse.linalg import cg, gmres, lsqr

class Sloc:

    def __init__(self, with_bias=False):
        self.prob = 0.5
        self.nmasks = 200
        self.with_bias = with_bias


    def gen_mask_resp(self, run_model, input_ids, target, is_return_logits=False):
        
        assert not is_return_logits
        device = input_ids.device
        def rmodel(inp):
            vals = run_model(inp)
            return torch.softmax(vals, dim=1)

        base = 0 #rmodel(torch.tensor([[]], device=device))[0,target].tolist()

        ntoks = input_ids.shape[1]
        masks = (torch.rand((self.nmasks, ntoks)) < self.prob)
        masks = masks[masks.sum(dim=1) > 1, :]
        responses = []
        linput = input_ids[0].cpu().tolist()
        for idx in range(masks.shape[0]):            
            imask = masks[idx].tolist()
            pert = [[tok for tok, lit in zip(linput, imask) if lit]]
            tpert = torch.tensor(pert, device=device)
            out = rmodel(tpert)
            resp = out[0, target].tolist()[0] - base
            responses.append(resp)

        return masks, torch.Tensor(responses)

    @torch.no_grad()
    def run(self, run_model, input_ids, target):
        sys.stdout.flush()
        masks, resps = self.gen_mask_resp(run_model, input_ids, target)

        Y = resps
        X = masks * 1.0
        if self.with_bias:
            masks = torch.concat([torch.ones(masks.shape[0],1), masks], dim=1)

        weights = torch.sqrt(torch.ones(1) / X.shape[0])
        Xw = X * weights
        XTXw = Xw.T @ Xw
        XTY = Xw.T @ Y
        
        c_magnitude = 0.01
        XTX = XTXw + torch.eye(XTXw.shape[0]) *  c_magnitude / XTXw.shape[0]
        bb, _info = gmres(XTX.numpy(), XTY.numpy())
        sal = torch.tensor(bb)    
        if self.with_bias:
            sal = sal[1:]
        #print("shape", sal.shape)
        return sal

