"""Parallel TCN/SSM with gated fusion and a shared linear torque head."""
import torch.nn as nn

from .tcn import TemporalConvNet
from .ssm import ResidualSSMBlockCumsum
from .fusion import GatedFusion


class ParallelTCNSSMLast(nn.Module):
    def __init__(self, input_dim=16, output_dim=4, num_channels=None,
                 kernel_size=7, dropout=0.4, ssm_branch_layers=1,
                 ssm_dropout=0.1, ssm_scale_init=0.1, ssm_expand=2, ssm_d_state=8):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64]
        d_model = num_channels[-1]
        self.tcn_branch = TemporalConvNet(input_dim, num_channels,
                                          kernel_size=kernel_size, dropout=dropout)
        self.ssm_branch = nn.ModuleList([
            ResidualSSMBlockCumsum(
                d_model=d_model, d_state=ssm_d_state, expand=ssm_expand,
                dropout=ssm_dropout, ssm_scale_init=ssm_scale_init
            ) for _ in range(ssm_branch_layers)
        ])
        self.tcn_proj = nn.Linear(input_dim, d_model)
        self.fusion = GatedFusion(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        h_tcn = self.tcn_branch(x.transpose(1, 2)).transpose(1, 2)
        h_ssm = self.tcn_proj(x)
        for ssm in self.ssm_branch:
            h_ssm = ssm(h_ssm)
        h_final = self.fusion(h_tcn[:, -1, :], h_ssm[:, -1, :])
        return self.head(h_final)


def create_model(config):
    if (config['model_name'] != 'parallel_tcn_ssm_last' or config['input_dim'] != 16
            or config['output_dim'] != 4 or config['fusion'] != 'gated'
            or config['head_type'] != 'linear'):
        raise ValueError('This release contains only the 16-input gated ParallelTCNSSMLast model')
    keys = ('input_dim', 'output_dim', 'num_channels', 'kernel_size', 'dropout',
            'ssm_branch_layers', 'ssm_dropout', 'ssm_scale_init', 'ssm_expand', 'ssm_d_state')
    return ParallelTCNSSMLast(**{key: config[key] for key in keys})
