from models.ae import AE
from models.base_units.blocks import BasicBlock, ResBlock


class AEU(AE):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=None, de_num_layers=None,residual=False, layer=4):
        super(AEU, self).__init__(input_size, in_planes, base_width, expansion, mid_num, latent_size, en_num_layers,
                                  de_num_layers, residual, layer)
        block = ResBlock if residual else BasicBlock
        self.de_block4 = block(1 * base_width * expansion,  2 * in_planes, 1, upsample=True,
                                    last_layer=True)

    def forward(self, x):
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        bottle_out = self.bottle_neck(en4)
        z, de1 = bottle_out['z'], bottle_out['out']

        de2 = self.de_block1(de1)
        de3 = self.de_block2(de2)
        de4 = self.de_block3(de3)
        x_hat, log_var = self.de_block4(de4).chunk(2, 1)

        return {'x_hat': x_hat, 'log_var': log_var, 'z': z,
                'features': [en1, en2, en3, en4, z, de1, de2, de3,de4][self.layer]}

