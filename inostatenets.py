import numpy as np
from torch import nn
import torch

class nonlin_ino_state_net(nn.Module):
    def __init__(self, nx, nu, ny, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh): 
        super(nonlin_ino_state_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.nx = nx
        self.net = simple_res_net(n_in=nx+np.prod(self.nu,dtype=int)+np.prod(self.ny,dtype=int), \
                                  n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, \
                                  activation=activation)

    def forward(self, x, u, eps=None):
        if eps==None:
            eps = torch.zeros((u.shape[0],np.prod(self.ny,dtype=int)),dtype=torch.float32)
        net_in = torch.cat([x,u.view(u.shape[0],-1),eps.view(u.shape[0],-1)],axis=1)            
        return self.net(net_in)