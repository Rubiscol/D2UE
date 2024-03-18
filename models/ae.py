import torch.nn as nn
from models.base_units.blocks import BasicBlock, BottleNeck,ResBlock


class AE(nn.Module):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=None, de_num_layers=None, residual=False, layer=4):
        super(AE, self).__init__()
        block = ResBlock if residual else BasicBlock
        self.fm = input_size // 16  # down-sample for 4 times. 2^4=16
        self.layer=layer

        if en_num_layers is None:
            en_num_layers = 1
        if de_num_layers is None:
            de_num_layers = 1

        self.en_block1 = block(in_planes, 1 * base_width * expansion, en_num_layers, downsample=True)

        self.en_block2 = block(1 * base_width * expansion, 2 * base_width * expansion, en_num_layers,
                                    downsample=True)
        self.en_block3 = block(2 * base_width * expansion, 4 * base_width * expansion, en_num_layers,
                                    downsample=True)
        self.en_block4 = block(4 * base_width * expansion, 4 * base_width * expansion, en_num_layers,
                                    downsample=True)

        self.bottle_neck = BottleNeck(4 * base_width * expansion, feature_size=self.fm, mid_num=mid_num,
                                      latent_size=latent_size)

        self.de_block1 = block(4 * base_width * expansion, 4 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block2 = block(4 * base_width * expansion, 2 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block3 = block(2 * base_width * expansion, 1 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block4 = block(1 * base_width * expansion, in_planes, de_num_layers, upsample=True,
                                    last_layer=True)

    def forward(self, x):
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        bottle_out = self.bottle_neck(en4)
        z, de4 = bottle_out['z'], bottle_out['out']

        de3 = self.de_block1(de4)
        de2 = self.de_block2(de3)
        de1 = self.de_block3(de2)
        x_hat = self.de_block4(de1)

        return {'x_hat': x_hat, 'z': z, 'features': [en1, en2, en3, en4, z, de4, de3, de2, de1][self.layer]}

