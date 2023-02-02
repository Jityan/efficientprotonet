import torch.nn as nn
import torch.nn.functional as F

def conv_block1d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(2)
    )

class Convnet1d(nn.Module):
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block1d(x_dim, hid_dim),
            conv_block1d(hid_dim, hid_dim),
            conv_block1d(hid_dim, z_dim),
        )
        self.weights_init(self.encoder)

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def weights_init(self,module):
        for m in module.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.uniform_(m.weight)
                m.bias.data.zero_()

class Convnet1d1c(nn.Module):
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block1d(x_dim, hid_dim),
            conv_block1d(hid_dim, hid_dim),
            conv_block1d(hid_dim, hid_dim),
            conv_block1d(hid_dim, hid_dim),
            conv_block1d(hid_dim, hid_dim),
            conv_block1d(hid_dim, z_dim),
        )
        self.weights_init(self.encoder)

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def weights_init(self,module):
        for m in module.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.uniform_(m.weight)
                m.bias.data.zero_()

class DistillKL(nn.Module):

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T
    
    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False)*(self.T**2)/y_s.shape[0]
        return loss
