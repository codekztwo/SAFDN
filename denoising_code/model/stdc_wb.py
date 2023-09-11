import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvX(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False, act='relu') -> None:
        super().__init__()
        self.act = act

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              padding=kernel_size // 2, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        if self.act == 'leaky':
            x = F.leaky_relu(x)
        else:
            x = F.relu(x)

        return x


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1) -> None:
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."

        self.conv_list = []
        self.stride = stride
        n = out_planes // 2

        if stride == 2:
            self.adv_layer = nn.Sequential(
                nn.Conv2d(n, n, kernel_size=3, stride=2, padding=1, groups=n, bias=False),
                nn.BatchNorm2d(n)
            )

            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )

            stride = 1

        for idx in range(block_num):
            if idx == 0:
                conv_layer = ConvX(in_planes, n, kernel_size=1)
            elif idx == 1 and block_num == 2:
                conv_layer = ConvX(n, n, stride=stride)
            elif idx == 1 and block_num > 2:
                conv_layer = ConvX(n, out_planes // 4, stride=stride)
            elif idx < (block_num - 1):
                conv_layer = ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1)))
            else:
                conv_layer = ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx)))

            self.conv_list.append(conv_layer)

        self.conv_list = nn.Sequential(*self.conv_list)

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.adv_layer(conv(out))
            else:
                out = conv(out)

            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        out = torch.concat(out_list, dim=1) + x

        return out


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1) -> None:
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."

        self.conv_list = []
        self.stride = stride
        n = out_planes // 2

        if stride == 2:
            self.adv_layer = nn.Sequential(
                nn.Conv2d(n, n, kernel_size=3, stride=(1, 2), padding=1, groups=n, bias=False),
                nn.BatchNorm2d(n)
            )

            self.skip = nn.AvgPool2d(kernel_size=3, stride=(1, 2), padding=1)

            stride = 1

        for idx in range(block_num):
            if idx == 0:
                conv_layer = ConvX(in_planes, n, kernel_size=1)
            elif idx == 1 and block_num == 2:
                conv_layer = ConvX(n, n, stride=stride)
            elif idx == 1 and block_num > 2:
                conv_layer = ConvX(n, out_planes // 4, stride=stride)
            elif idx < (block_num - 1):
                conv_layer = ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1)))
            else:
                conv_layer = ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx)))

            self.conv_list.append(conv_layer)

        self.conv_list = nn.Sequential(*self.conv_list)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        out = None
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.adv_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)

            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)

        out_list.insert(0, out1)
        out = torch.concat(out_list, dim=1)

        return out


class PreBlock(nn.Module):

    def __init__(self, in_channels, n, stride=(1, 2)):
        super(PreBlock, self).__init__()

        self.branch1 = BasicConv2d(in_channels, n, kernel_size=(7, 3), padding=(3, 1), stride=stride)
        self.branch2 = BasicConv2d(in_channels, n, kernel_size=3, padding=1, stride=stride)
        self.branch3 = BasicConv2d(in_channels, n, kernel_size=3, dilation=(2, 2), padding=2, stride=stride)
        self.branch4 = BasicConv2d(in_channels, n, kernel_size=(3, 7), padding=(1, 3), stride=stride)
        # self.conv = BasicConv2d(n * 4, n, kernel_size=1, padding=1, stride=(1, 1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        output = torch.cat([branch1, branch2, branch3, branch4], 1)
        # output = self.conv(output)

        return output


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class STDCNet(nn.Module):
    def __init__(self, base=64, layers=[4, 5, 3], block_num=4, type="cat") -> None:
        super().__init__()

        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck

        self.layes = layers
        self.feat_channels = [base // 2, base, base * 2, base * 4, base * 8]
        self.features = self._make_layers(base, layers, block_num, block)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1: 2])
        self.x8 = nn.Sequential(self.features[2: 2 + self.layes[0]])
        self.x16 = nn.Sequential(self.features[2 + self.layes[0]: 2 + sum(self.layes[0: 2])])
        self.x32 = nn.Sequential(self.features[2 + sum(self.layes[0: 2]): 2 + sum(self.layes)])

        self.init_params()

    def _make_layers(self, base, layers, block_num, block):
        # #                                                                               原来的stage1 && stage2
        # features = [ConvX(2, base // 2, kernel_size=3, stride=(1, 2)),
        #             ConvX(base // 2, base, kernel_size=3, stride=(1, 2))]

        features = [PreBlock(2, 8, stride=(1, 1)), PreBlock(32, 16, stride=(1, 2))]     # stage1 && stage2
        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 2, block_num, 2))
                elif j == 0:
                    features.append(
                        block(base * int(math.pow(2, i)), base * int(math.pow(2, i + 1)), block_num, 2))
                else:
                    features.append(
                        block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 1)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)

        return feat2, feat4, feat8, feat16, feat32

    def init_weight(self, pretrain_model):
        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()

        for k, v in state_dict.items():
            self_state_dict.update({k: v})

        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    from thop import profile

    model = STDCNet(base=64, layers=[2, 2, 2])
    model.eval()
    model.to('cuda')
    d_input = torch.zeros((1, 2, 32, 400)).cuda()
    y = model(d_input)
    for t in y:
        print(t.shape)
    print(model.feat_channels)
    # flops, params = profile(model, (d_input, ))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))