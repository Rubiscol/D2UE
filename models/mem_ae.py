from models.ae import AE
from models.base_units.blocks import MemBottleNeck


class MemAE(AE):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=None, de_num_layers=None, mem_size=25, shrink_thres=0.0025,residual=False, layer=4):
        super(MemAE, self).__init__(input_size, in_planes, base_width, expansion, mid_num, latent_size, en_num_layers,
                                    de_num_layers, residual, layer)

        self.bottle_neck = MemBottleNeck(4 * base_width * expansion, feature_size=self.fm, mid_num=mid_num,
                                         latent_size=latent_size, mem_size=mem_size, shrink_thres=shrink_thres)

    def forward(self, x):
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        bottle_out = self.bottle_neck(en4)
        z, z_hat, att, de1 = bottle_out['z'], bottle_out['z_hat'], bottle_out['att'], bottle_out['out']

        de2 = self.de_block1(de1)
        de3 = self.de_block2(de2)
        de4 = self.de_block3(de3)
        x_hat = self.de_block4(de4)

        return {'x_hat': x_hat, 'z': z, 'z_hat': z_hat, 'att': att,
                'features': [en1, en2, en3, en4, z, de1, de2, de3,de4][self.layer]}
