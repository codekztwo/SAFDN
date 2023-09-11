import torch
import torch.nn as nn
import torch.nn.functional as F

from denoising_code.model.stdc import STDCNet, ConvX
# from stdc import STDCNet, ConvX

class SPPM(nn.Module):
    """Simple Pyramid Pooling Module
    """
    def __init__(self, in_channels, inter_channels, out_channels, bin_sizes) -> None:
        """
        :param in_channels: int, channels of input feature
        :param inter_channels: int, chennels of mid conv
        :param out_channels: int, channels of output feature
        :param bin_sizes: list, avg pool size of 3 features
        """
        super().__init__()

        self.stage1_pool = nn.AdaptiveAvgPool2d(output_size=bin_sizes[0])
        self.stage1_conv = ConvX(in_channels, inter_channels, kernel_size=1)

        self.stage2_pool = nn.AdaptiveAvgPool2d(output_size=bin_sizes[1])
        self.stage2_conv = ConvX(in_channels, inter_channels, kernel_size=1)

        self.stage3_pool = nn.AdaptiveAvgPool2d(output_size=bin_sizes[2])
        self.stage3_conv = ConvX(in_channels, inter_channels, kernel_size=1)

        self.conv_out = ConvX(inter_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h, w =  x.size()[2:]

        f1 = self.stage1_pool(x)
        f1 = self.stage1_conv(f1)
        f1 =  F.interpolate(f1, (h, w), mode='bilinear', align_corners=False)
    
        f2 = self.stage2_pool(x)
        f2 = self.stage2_conv(f2)
        f2 =  F.interpolate(f2, (h, w), mode='bilinear', align_corners=False)

        f3 = self.stage3_pool(x)
        f3 = self.stage3_conv(f3)
        f3 =  F.interpolate(f3, (h, w), mode='bilinear', align_corners=False)

        x = self.conv_out(f1 + f2 + f3)

        return x


class UAFM(nn.Module):    #原统一注意力机制模块
    """Unified Attention Fusion Modul
    """
    def __init__(self, low_chan, hight_chan, out_chan, u_type='sp') -> None:
        """
        :param low_chan: int, channels of input low-level feature
        :param hight_chan: int, channels of input high-level feature
        :param out_chan: int, channels of output faeture
        :param u_type: string, attention type, sp: spatial attention, ch: channel attention
        """
        super().__init__()
        self.u_type = u_type

        if u_type == 'sp':
            self.conv_atten = nn.Sequential(
                ConvX(4, 2, kernel_size=3),
                nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(1),
                )
        else:
            self.conv_atten = nn.Sequential(
                ConvX(4 * hight_chan, hight_chan // 2,  kernel_size=1, bias=False, act="leaky"),
                nn.Conv2d(hight_chan // 2, hight_chan, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(hight_chan),
            )

        self.conv_low = ConvX(low_chan, hight_chan, kernel_size=3, padding=1, bias=False)
        self.conv_out = ConvX(hight_chan, out_chan, kernel_size=3, padding=1, bias=False)

    def _spatial_attention(self, x):
        """
        :param x: tensor, feature
        :return x: tensor, fused feature
        """
        mean_value = torch.max(x, dim=1, keepdim=True)[0]
        max_value = torch.mean(x, dim=1, keepdim=True)

        value = torch.concat([mean_value, max_value], dim=1)

        return value

    def _channel_attention(self, x):
        """
        :param x: tensor, feature
        :return x: tensor, fused feature
        """
        avg_value = F.adaptive_avg_pool2d(x, 1)
        max_value = torch.max(torch.max(x, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
        value = torch.concat([avg_value, max_value], dim=1)

        return value
    def forward(self, x_high, x_low):
        """
        :param x_high: tensor, high-level feature
        :param x_low: tensor, low-level feature
        :return x: tensor, fused feature
        """
        h, w =  x_low.size()[2:]

        x_low = self.conv_low(x_low)
        x_high = F.interpolate(x_high, (h, w), mode='bilinear', align_corners=False)

        if self.u_type == 'sp':
            atten_high = self._spatial_attention(x_high)
            atten_low = self._spatial_attention(x_low)
        else:
            atten_high = self._channel_attention(x_high)
            atten_low = self._channel_attention(x_low)

        atten = torch.concat([atten_high, atten_low], dim=1)
        atten = torch.sigmoid(self.conv_atten(atten))

        x = x_high * atten + x_low * (1 - atten)
        x = self.conv_out(x)

        return x



class SegHead(nn.Module):
    """FLD Decoder
    """
    def __init__(self, bin_sizes, decode_chans):
        """
        :param bin_sizes: list, avg pool size of 3 features
        :param decode_chans: list, channels of decoder feature size
        """
        super().__init__()

        self.sppm = SPPM(512, decode_chans[0], decode_chans[0], bin_sizes) #1024    # stdc1 use [128, 64, 32]
        self.uafm1 = UAFM(256, decode_chans[0], decode_chans[1])    #512
        self.uafm2 = UAFM(128, decode_chans[1], decode_chans[2])    #256

    def forward(self, x):
        # x8, x16, x32
        sppm_feat = self.sppm(x[-1])
  
        merge_feat1 = self.uafm1(sppm_feat, x[1])
        merge_feat2 = self.uafm2(merge_feat1, x[0])

  
        return [sppm_feat, merge_feat1, merge_feat2]


class SegClassifier(nn.Module):
    """Classification Layer
    """
    def __init__(self, in_chan, mid_chan, n_classes) -> None:
        """
        :param in_chan: int, channels of input feature
        :param mid_chan: int, channels of mid conv
        :param n_classes: int, number of classification
        """
        super().__init__()
        self.conv = ConvX(in_chan, mid_chan, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)

        return x


class PPLiteSeg(nn.Module):
    def __init__(self, n_classes, t='stdc1') -> None:
        """
        :param n_classes: int, number of classification
        :param t: string, backbone type, stdc1/stdc2
        """
        super().__init__()

        if t == 'stdc1':
            layers=[2, 2, 2]
        else:
            layers=[4, 5, 3]
        backbone_indices=[2, 3, 4]
        bin_sizes=[1, 2, 4]
        decode_chans = [128, 64, 32]    # stdc2 use [128, 96, 64]  stdc1 use [128, 64, 32]
  
        self.backbone = STDCNet(64, layers)
        self.backbone_indices = backbone_indices
        self.seg_head = SegHead(bin_sizes, decode_chans)

        self.classifer = []
        # for chan in decode_chans:
        #     cls = SegClassifier(chan, 64, n_classes)
        #     self.classifer.append(cls)
        self.classifer = nn.Sequential(SegClassifier(decode_chans[-1],64,n_classes))

    def forward(self, x):
        h, w = x.size()[2:]

        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        feats_selected = [feats_backbone[i] for i in self.backbone_indices]

        head_out = self.seg_head(feats_selected)
        if self.training:
            outs = F.interpolate(head_out[-1], (h, w), mode='bilinear', align_corners=False)
            outs = self.classifer[-1](outs)
        else:
            outs = F.interpolate(head_out[-1], (h, w), mode='bilinear', align_corners=False)
            outs = self.classifer[-1](outs)
            outs = torch.softmax(outs, dim=1)

        return outs


if __name__ == '__main__':
    from thop import profile
    model = PPLiteSeg(4)
    # model.load_state_dict(torch.load('checkpoints_PPliteseg_tezhen/model_epoch16_mIoU=73.1.pth'))

    model.load_state_dict(torch.load('/home/tony/github/SAFDN/checkpoints_PPliteseg_both/model_epoch43_mIoU=90.4.pth'))
    model.eval()
    model.to('cuda')
    # x = torch.zeros((1, 2, 32, 400)).cuda()
    x = torch.zeros((1, 2, 32, 720)).cuda()


    # y = model(x)
    # print(len(y))
    # print(y.shape)

    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))




