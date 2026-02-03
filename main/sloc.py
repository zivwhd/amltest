
## input_ids tensor([[  101,  1045,  2001,  3110,  2428, 11587,  1998,  2091,  2058,  2054, 2026,  3611,  2056,   102]], device='cuda:0')
#logits tensor([[ 6.7980, -1.6343, -1.4902, -0.8498, -0.9241, -1.8199]], device='cuda:0')
#attr-scores: tensor([ 0.0000, -0.0239,  0.0182,  0.3137,  0.0000,  0.7476,  0.2085,  0.0019, 0.0000, -0.0074, -0.0612, -0.2065,  0.0054,  0.1682], device='cuda:0')
#attr-scores-shape: torch.Size([14])
#logits: tensor([[ 6.7980, -1.6343, -1.4902, -0.8498, -0.9241, -1.8199]], device='cuda:0')

import torch
import sys
from scipy.sparse.linalg import cg, gmres, lsqr
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

class Sloc:

    def __init__(self, with_bias=False, l2_weight=0.01, mode="linear", baseline_token=None, pwidth=None):
        self.prob = 0.5
        self.nmasks = 200
        self.with_bias = with_bias
        self.l2_weight = 0.01
        self.mode = mode
        self.baseline_token = baseline_token
        self.pwidth = pwidth


    def gen_part_mask(self, shape, pwidth, prob):
        flip_prob = 1.0 / pwidth
        flip = (torch.rand(shape) < flip_prob)*1
        idx = flip.cumsum(dim=1) + (torch.arange(flip.shape[0]) * flip.shape[1]).unsqueeze(1)
        rnd = torch.rand(flip.numel())
        rnd = rnd[idx.flatten()].reshape(flip.shape)
        return rnd < prob

    def gen_mask_resp(self, run_model, input_ids, target, is_return_logits=False):
        
        assert not is_return_logits
        device = input_ids.device
        def rmodel(inp):
            vals = run_model(inp)
            return torch.softmax(vals, dim=1)

        base = 0 #rmodel(torch.tensor([[]], device=device))[0,target].tolist()

        ntoks = input_ids.shape[1]
        if self.pwidth is None:
            masks = (torch.rand((self.nmasks, ntoks)) < self.prob)
        else:
            masks = self.gen_part_mask((self.nmasks, ntoks), self.pwidth, self.prob)
        linput = input_ids[0].cpu().tolist()

        if self.baseline_token is None:
            masks = masks[masks.sum(dim=1) > 1, :]
        else:
            baseline = [[(self.baseline_token) for tok in linput]]
            baseline = torch.tensor(baseline, device=device)
            out = rmodel(baseline)
            base = out[0, target].tolist()[0]
            
        responses = []
        
        for idx in range(masks.shape[0]):            
            imask = masks[idx].tolist()
            if self.baseline_token:
                pert = [[(tok if lit else self.baseline_token) for tok, lit in zip(linput, imask)]]
            else:
                pert = [[tok for tok, lit in zip(linput, imask) if lit]]
            print("EE", pert, len(pert))
            tpert = torch.tensor(pert, device=device)
            out = rmodel(tpert)
            resp = out[0, target].tolist()[0] - base
            responses.append(resp)

        return masks, torch.Tensor(responses)

    def run(self, run_model, input_ids, target):
        if self.mode == "linear":
            return self.run_linear(run_model, input_ids, target)
        elif self.mode == "logistic":
            return self.run_logistic(run_model, input_ids, target)
        else:
            raise Exception(f"bad mode {self.mode}")
        
    @torch.no_grad()
    def run_linear(self, run_model, input_ids, target):
        sys.stdout.flush()
        masks, resps = self.gen_mask_resp(run_model, input_ids, target)

        Y = resps
        X = masks * 1.0
        if self.with_bias:
            X = torch.concat([torch.ones(X.shape[0],1), X], dim=1)

        weights = torch.sqrt(torch.ones(1) / X.shape[0])
        Xw = X * weights
        XTXw = Xw.T @ Xw
        XTY = Xw.T @ Y
        
        c_magnitude = self.l2_weight
        XTX = XTXw + torch.eye(XTXw.shape[0]) *  c_magnitude / XTXw.shape[0]
        bb, _info = gmres(XTX.numpy(), XTY.numpy())
        sal = torch.tensor(bb)    
        if self.with_bias:
            sal = sal[1:]
        #print("shape", sal.shape)
        return sal

    @torch.no_grad()
    def run_logistic(self, run_model, input_ids, target):
        sys.stdout.flush()
        masks, resps = self.gen_mask_resp(run_model, input_ids, target)

        Y = resps
        X = masks * 1.0
        
        add_bias = self.with_bias
        ###
        #model = LogisticRegression(
        #    penalty='l2', C=self.l2_weight, solver='lbfgs')  # L2 regularization
        #model.fit(X.numpy(), Y.numpy())
        if add_bias:
            X = torch.concat([torch.ones(X.shape[0],1), X], dim=1)
        
        model = sm.GLM(Y.numpy(), X.numpy(), family=sm.families.Binomial())
        results = model.fit()        
        sal = torch.tensor(results.params, dtype=torch.float32)
        print(sal)
        print(sal.shape)
        if add_bias:
            sal = sal[1:]

        return sal

