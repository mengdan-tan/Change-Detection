import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools
from einops import rearrange

import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from models.extra_model import SR


###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = ResNet(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'base_transformer_pos_s4':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos='learned')

    elif args.net_G == 'base_transformer_pos_s4_dd8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos='learned', enc_depth=1, dec_depth=8)

    elif args.net_G == 'base_transformer_pos_s4_dd8_dedim8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)

    elif args.net_G == 'base_transformer_pos_s4_dd8_resnet50':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=5,
                               backbone='resnet50', with_pos='learned', enc_depth=1, dec_depth=8)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################
class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        # expand是何参数？？
        expand = 1

        # 使用了预训练权重
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')  # 双线性插值

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        # layers为输入通道数
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    # 两张图片输入CNN_backbone：Resnet
    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        # 提取出特征后
        x = torch.abs(x1 - x2)  # 差的绝对值，为何要作差？哪里用到了？
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x  # 输出的是什么？？

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)  # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)  # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)  # 1/4 in=256,out=256
        else:
            x = x_8
        # output layers
        # self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)
        x = self.conv_pred(x)  # 大小不变，通道数变成32
        return x


# ###############################################################################
# # BIT的主干网络中加入ASPP空洞卷积池化金字塔？？
# # _forward_semantic_tokens中将Spatial Attention 改进一下？
# # 考虑网络的复杂度
# ###############################################################################
# class ChannelAttention(nn.Module):
#     def __init__(self, inplanes):
#         super(ChannelAttention, self).__init__()
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # 通道注意力，即两个全连接层连接
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels=inplanes, out_channels=inplanes // 16, kernel_size=1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=inplanes // 16, out_channels=inplanes, kernel_size=1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_out = self.fc(self.max_pool(x))
#         avg_out = self.fc(self.avg_pool(x))
#         # 最后输出的注意力应该为非负
#         out = self.sigmoid(max_out + avg_out)
#         return out
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # 压缩通道提取空间信息
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         # 经过卷积提取空间注意力权重
#         x = torch.cat([max_out, avg_out], dim=1)
#         out = self.conv1(x)
#         # 输出非负
#         out = self.sigmoid(out)
#         return out
#
# # 通常在backbone一层卷积之后插入ASPP，但是如此的话就不符合用轻量级CNN backbone
# class ASPP(nn.Module):
#     def __init__(self, in_channel=512, depth=256):
#         super(ASPP,self).__init__()
#         self.mean = nn.AdaptiveAvgPool2d((1, 1))
#         self.conv = nn.Conv2d(in_channel, depth, 1, 1)
#         self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
#         # 不同空洞率的卷积
#         self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
#         self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
#         self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
#         self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
#
#     def forward(self, x):
#         size = x.shape[2:]
#      # 池化分支
#         image_features = self.mean(x)
#         image_features = self.conv(image_features)
#         image_features = F.upsample(image_features, size=size, mode='bilinear')
#      # 不同空洞率的卷积
#         atrous_block1 = self.atrous_block1(x)
#         atrous_block6 = self.atrous_block6(x)
#         atrous_block12 = self.atrous_block12(x)
#         atrous_block18 = self.atrous_block18(x)
#         # 汇合所有尺度的特征
#         x = torch.cat([image_features, atrous_block1, atrous_block6,atrous_block12, atrous_block18], dim=1)
#         # 利用1X1卷积融合特征输出
#         x = self.conv_1x1_output(x)
#         return x


#
# # #############################################################################
# # BASE_Transformer的Semantic_token中加入CA attention
# # #############################################################################
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
#
#
# # 输入输出维度相同
# class CoordAtt(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#
#         mip = max(8, inp // reduction)
#
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#
#         n, c, h, w = x.size()
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)
#
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#
#         out = identity * a_w * a_h
#
#         return out
# # #############################################################################


class BASE_Transformer(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """

    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer, self).__init__(input_nc, output_nc, backbone=backbone,
                                               resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        # # CA attention
        # self.ca = CoordAtt(inp=32,oup=32)   # 输入输出通道数均为32
        self.SiamSR = SR(32)  # 输入通道数为32

        self.token1_conv = nn.Conv2d(in_channels=2 * token_len, out_channels=token_len, kernel_size=1)
        self.token2_conv = nn.Conv2d(in_channels=2 * token_len, out_channels=token_len, kernel_size=1)

        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans

        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2 * dim

        self.with_pos = with_pos
        # position信息
        if with_pos is 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, 32))
        decoder_pos_size = 256 // 4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder = nn.Parameter(torch.randn(1, 32,
                                                                  decoder_pos_size,
                                                                  decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                                                      heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim,
                                                      dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        # # 空间注意力机制，提取空间信息，得到token,是否过于草率了？
        # # semantic_token类似于分词器
        # x=self.ca(x)

        # x0=self.SiamSR(x)
        spatial_attention = self.conv_a(x)  # 1*1 conv,输出通道数为token_len
        # reshape
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()  # reshape x
        # 将 spatial_attention 作为权重对 x 进行加权求和操作，得到 tokens
        # 这一步调试的时候出错，是由于显存不足计算不出来
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        # 函数返回 tokens，它的形状为 [batch_size, self.token_len, channels]。每个样本在 self.token_len 个语义令牌上进行加权聚合后得到一个特征向量
        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode is 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode is 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')  # reshape
        return tokens

    # 判断位置信息
    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h, w, b, l, c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # input:(1,3,256,256)
        # forward backbone resnet
        # resnet提取特征
        x1 = self.forward_single(x1)  # (1,32,64,64)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)  # (1,4,32)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens_2 = torch.cat([token2, token1], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)  # (1,8,32)
            self.tokens2 = self._forward_transformer(self.tokens_2)  # (1,8,32)
            tokens1_1, tokens1_2 = self.tokens.chunk(2, dim=1)
            tokens2_2, tokens2_1 = self.tokens_2.chunk(2, dim=1)
            token1_ = torch.cat([tokens1_1, tokens2_1], dim=1)
            token2_ = torch.cat([tokens1_2, tokens2_2], dim=1)
            token1=self.token1_conv(token1_)
            token2=self.token1_conv(token2_)

        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)  # (1,32,64,64)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)

        # feature differencing，上采样
        # 输入输出均为(1,32,64,64)
        x = torch.abs(x1 - x2)  # 差的绝对值，(1,32,64,64)

        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)  # (1,32,256,256)
        # forward small cnn
        # 经过浅层CNN，得到最后二值图像
        x = self.classifier(x)  # (1,2,256,256)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x
