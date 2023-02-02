# The code is modified from efficientnet pytorch
# (https://github.com/narumiruna/efficientnet-pytorch)

import math
import torch
import torch.nn as nn
#import torch.nn.functional as F

params = {
    'efficientnet_b0': (1.0, 1.0, 224, 0.2)
}

model_urls = {
    'efficientnet_b0': 'https://www.dropbox.com/s/9wigibun8n260qm/efficientnet-b0-4cfa50.pth?dl=1',
}

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_c, out_c, kernel_size, stride=1, groups=1):
        padding = self.get_padding(kernel_size, stride)
        super(ConvBnRelu, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            Swish()
        )
    
    def get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p//2, p-p//2, p//2, p-p//2]

class SqueezeExcitation(nn.Module):
    def __init__(self, in_c, reduce_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, reduce_dim, 1),
            Swish(),
            nn.Conv2d(reduce_dim, in_c, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_c, out_c, expand_ratio, kernel_size, stride, reduction_ratio=4, drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_c == out_c and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_c * expand_ratio
        reduce_dim = max(1, int(in_c/reduction_ratio))
        layers = []

        if in_c != hidden_dim:
            layers += [ConvBnRelu(in_c, hidden_dim, 1)]
        
        layers += [
            ConvBnRelu(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, reduce_dim),
            nn.Conv2d(hidden_dim, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ]
        self.conv = nn.Sequential(*layers)
    
    def drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()
    
    def forward(self, x):
        if self.use_residual:
            return x + self.drop_connect(self.conv(x))
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=1000):
        super(EfficientNet, self).__init__()
        settings = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3]
        ]

        out_channels = self.round_filters(32, width_mult)
        features = [ConvBnRelu(3, out_channels, 3, stride=2)]

        in_channels = out_channels
        for t, c, n, s, k in settings:
            out_channels = self.round_filters(c, width_mult)
            repeats = self.round_repeats(n, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                in_channels = out_channels
        last_channels = self.round_filters(1280, width_mult)
        features += [ConvBnRelu(in_channels, last_channels, 1)]
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )

        self.initialize_weights()

    def round_repeats(self, repeats, depth_mult):
        if depth_mult == 1.0:
            return repeats
        return int(math.ceil(depth_mult * repeats))
    
    def round_filters(self, filters, width_mult):
        if width_mult == 1.0:
            return filters
        new_value = max(width_mult, int(filters + width_mult / 2) // width_mult * width_mult)
        if new_value < 0.9 * filters:
            new_value += width_mult
        return new_value
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, is_feat=False):
        x = self.features(x)
        x = x.mean([2, 3]) # flatten
        if is_feat:
            return x
        x = self.classifier(x)
        return x


def returnOriModel():
    try:
        from torch.hub import load_state_dict_from_url
    except ImportError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url
    width_mult, depth_mult, _, dropout_rate = params['efficientnet_b0']
    model = EfficientNet(width_mult, depth_mult, dropout_rate)
    state_dict = load_state_dict_from_url(model_urls['efficientnet_b0'], progress=True)
    model.load_state_dict(state_dict, strict=False)
    return model

def returnCusModel(cls=1000):
    width_mult, depth_mult, _, dropout_rate = params['efficientnet_b0']
    model = EfficientNet(width_mult, depth_mult, dropout_rate, num_classes=cls)
    return model

if __name__ == "__main__":
    print("---EfficientNet---")