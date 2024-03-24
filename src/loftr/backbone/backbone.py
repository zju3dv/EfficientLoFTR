import torch.nn as nn
import torch.nn.functional as F
from .repvgg import create_RepVGG

class RepVGG_8_1_align(nn.Module):
    """
    RepVGG backbone, output resolution are 1/8 and 1.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        backbone = create_RepVGG(False)

        self.layer0, self.layer1, self.layer2, self.layer3 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3

        for layer in [self.layer0, self.layer1, self.layer2, self.layer3]:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer0(x) # 1/2
        for module in self.layer1:
            out = module(out) # 1/2
        x1 = out
        for module in self.layer2:
            out = module(out) # 1/4
        x2 = out
        for module in self.layer3:
            out = module(out) # 1/8
        x3 = out
                
        return {'feats_c': x3, 'feats_f': None, 'feats_x2': x2, 'feats_x1': x1}
