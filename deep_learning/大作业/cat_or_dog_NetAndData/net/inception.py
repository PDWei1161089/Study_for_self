import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):  # 模型输出类别数量为1000,
        # 允许在模型的辅助层上生成额外的输出， 不允许在处理图像数据之前对其进行变换
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        # BasicConv2d, InceptionA/B/C/D/E在下文均有定义
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)  # 第一层a卷积输入3通道， 输出32通道， 卷积核3*3， 卷积步长为2
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)  # 第二层a卷积输入32通道，输出32通道，卷积步长为3
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)  # 第二层b卷积输入为32通道，输出为64通道，卷积核3*3，
        # 填充像素为1（保证输入与输出大小一致）
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)  # 第三层b卷积输入为64通道，输出为80通道，卷积核1*1
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)  # 第四层a卷积输入为80通道，输出为192通道，卷积核为3*3
        self.Mixed_5b = InceptionA(192, pool_features=32)  # 第5层b卷积输入通道数为192，最大池化层输出为32通道（卷积核）
        self.Mixed_5c = InceptionA(256, pool_features=64)  # 第5层c卷积输入通道数为256，最大池化层输出为64通道（卷积核）
        self.Mixed_5d = InceptionA(288, pool_features=64)  # 第5层d卷积输入通道数为288，最大池化层输出为64通道（卷积核）
        self.Mixed_6a = InceptionB(288)  # 第6层a卷积有288个卷积核
        self.Mixed_6b = InceptionC(768, channels_7x7=128)  # 第6层b卷积输入通道数为768，以7*7卷积核来输出128通道
        self.Mixed_6c = InceptionC(768, channels_7x7=160)  # 第6层c卷积输入通道数为768，以7*7卷积核来输出160通道
        self.Mixed_6d = InceptionC(768, channels_7x7=160)  # 第6层d卷积输入通道数为768，以7*7卷积核来输出160通道
        self.Mixed_6e = InceptionC(768, channels_7x7=192)  # 第6层e卷积输入通道数为768，以7*7卷积核来输出192通道
        if aux_logits:  # 如果允许在模型的辅助层上生成额外的输出
            self.AuxLogits = InceptionAux(768, num_classes)  # 该类在下面有定义
        self.Mixed_7a = InceptionD(768)  # 第7层a卷积输入通道数为768
        self.Mixed_7b = InceptionE(1280)  # 第7层b卷积输入通道数为1280
        self.Mixed_7c = InceptionE(2048)  # 第7层c卷积输入通道数为2048
        self.fc = nn.Linear(2048, num_classes)  # 将输入的2048个数据进行处理

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):  # 如果输入数据为Conv2d或Linear的实例
                import scipy.stats as stats  # 调用统计数据库
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1  # 检查m是否为标准差，否则为0.1
                X = stats.truncnorm(-2, 2, scale=stddev)  # numpy中函数，正态分布，下界-2，上界2，方差为stddev
                values = torch.Tensor(X.rvs(m.weight.data.numel()))  # m的权重参数矩阵的元素总数，在rvs方法下填充成一个新的张量
                values = values.view(m.weight.data.size())  # 以m的权重参数矩阵的结构重构values
                m.weight.data.copy_(values)  # 将values深拷贝到m上
            elif isinstance(m, nn.BatchNorm2d):  # 如果m是BatchNorm2d的实例
                m.weight.data.fill_(1)  # 将所有元素设置为1
                m.bias.data.zero_()  # 将m的重置为所有元素均为0

    def forward(self, x):  # 定义前向传播函数
        if self.transform_input:  # 如果允许在处理图像数据之前对其进行变换
            x = x.clone()  # 创建一个x的深拷贝
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5  # 将第一个通道的数据归一化处理
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5  # 将第二个通道的数据归一化处理
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5  # 将第三个通道的数据归一化处理
        # 299 x 299 x 3  ## 输入数据大小 229*229*3
        x = self.Conv2d_1a_3x3(x)  # 进行第一层a卷积
        # 149 x 149 x 32  ## 输出数据大小 149*149*32
        x = self.Conv2d_2a_3x3(x)  # 进行第二层a卷积
        # 147 x 147 x 32  ## 输出数据大小 147*147*32
        x = self.Conv2d_2b_3x3(x)  # 进行第二层b卷积
        # 147 x 147 x 64  ## 输出数据大小 147*147*64
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 在3*3的最大池化核下进行操作，移动步长为2
        # 73 x 73 x 64  ## 输出数据大小为73*73*64
        x = self.Conv2d_3b_1x1(x)  # 进行第三层b卷积
        # 73 x 73 x 80  ## 输出数据大小为73*73*80
        x = self.Conv2d_4a_3x3(x)  # 进行第四层a卷积
        # 71 x 71 x 192  ## 输出数据大小为71*71*192
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 在3*3的最大池化核下进行操作，移动步长为2
        # 35 x 35 x 192  ## 输出数据大小为35*35*192
        x = self.Mixed_5b(x)  # 进行第五层b卷积
        # 35 x 35 x 256  ## 输出数据大小为35*35*256
        x = self.Mixed_5c(x)  # 进行第五层c卷积
        # 35 x 35 x 288  ## 输出数据大小为35*35*288
        x = self.Mixed_5d(x)  # 进行第五层d卷积
        # 35 x 35 x 288  ## 输出数据大小为35*35*288
        x = self.Mixed_6a(x)  # 进行第六层a卷积
        # 17 x 17 x 768  ## 输出数据大小为17*17*768
        x = self.Mixed_6b(x)  # 进行第六层b卷积
        # 17 x 17 x 768  ## 输出数据大小为17*17*768
        x = self.Mixed_6c(x)  # 进行第六层c卷积
        # 17 x 17 x 768  ## 输出数据大小为17*17*768
        x = self.Mixed_6d(x)  # 进行第六层d卷积
        # 17 x 17 x 768  ## 输出数据大小为17*17*768
        x = self.Mixed_6e(x)  # 进行第六层e卷积
        # 17 x 17 x 768  ## 输出数据大小为17*17*768
        if self.training and self.aux_logits:  # 允许在模型的辅助层上生成额外的输出，并且开启训练模式
            aux = self.AuxLogits(x)  # 进行InceptionAux卷积
        # 17 x 17 x 768  ## 输出数据大小为17*17*768
        x = self.Mixed_7a(x)  # 进行第七层a卷积
        # 8 x 8 x 1280  ## 输出数据大小为8*8*1280
        x = self.Mixed_7b(x)  # 进行第七层b卷积
        # 8 x 8 x 2048  ## 输出数据大小为8*8*2048
        x = self.Mixed_7c(x)  # 进行第七层c卷积
        # 8 x 8 x 2048  ## 输出数据大小为8*8*2048
        x = F.avg_pool2d(x, kernel_size=8)  # 二维平均池化操作，池化核尺寸为8*8
        # 1 x 1 x 2048  ## 输出数据大小为1*1*2048
        x = F.dropout(x, training=self.training)  # 按照伯努利分布随机置0
        # 1 x 1 x 2048  ## 输出数据大小为1*1*2048
        x = x.view(x.size(0), -1)  # 查看在第一维度下的数据大小
        # 2048  ## 输出数据大小为2048
        x = self.fc(x)  # 进行Linear全连接处理
        # 1000 (num_classes)  ## 输出数据大小为1000
        if self.training and self.aux_logits:  # 开启训练模式并且允许在模型的辅助层上生成额外的输出
            return x, aux  # 返回数据和InceptionAux卷积的结果
        return x  # 返回数据最终结果


class InceptionA(nn.Module):  # 定义InceptionA方法

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)  # 将数据进行第一轮卷积处理，输出通道数为64，卷积核为1*1

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)  # 第二轮第一次卷积处理，输出通道数为48， 卷积核大小为1*1
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)  # 第二轮第二次卷积处理，输出通道数为64，卷积核大小为5*5，边界填充2个像素

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)  # 第三轮第一次卷积处理，输出通道数为64，卷积核大小为1*1
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)  # 第三轮第二次卷积处理，输出通道数为96，卷积核大小为3*3，边界填充1个像素
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)  # 第三轮第三次卷积处理，输出通道数为96，卷积核大小为3*3，边界填充1个像素

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)  # 第四轮卷积处理，卷积核大小为1*1

    def forward(self, x):  # 定义前向传播函数
        branch1x1 = self.branch1x1(x)  # 第一轮卷积处理

        branch5x5 = self.branch5x5_1(x)  # 第二轮卷积处理
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)  # 第三轮卷积处理
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)  # 二维平均池化操作，卷积核大小3*3，移动步长为1，边界填充像素为1
        branch_pool = self.branch_pool(branch_pool)  # 进行二位平均池化操作

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]  # 设定输出值
        return torch.cat(outputs, 1)  # 按照branch5*5的维度将outputs中的张量进行拼接


class InceptionB(nn.Module):  # 定义InceptionB类

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)  # 第一轮卷积处理，输出通道数为384，卷积核大小为3*3，步长为2

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)  # 第二轮第一次卷积处理，输出通道数为64，卷积核大小为1*1
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)  # 第二轮第二次卷积处理，输出通道数为96，卷积核大小为3*3，边缘填充1个像素
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)  # 第二轮第三次卷积处理，输出通道数为96，卷积核大小为3*3，步长为2

    def forward(self, x):  # 定义前向传播函数
        branch3x3 = self.branch3x3(x)  # 进行第一轮卷积处理

        branch3x3dbl = self.branch3x3dbl_1(x)  # 进行第二轮卷积处理
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)  # 将x在池化核为3*3的池化层进行卷积处理，步长为2

        outputs = [branch3x3, branch3x3dbl, branch_pool]  # 设置输出值
        return torch.cat(outputs, 1)  # 以branch3*3dbl的维度将输出值进行拼接


class InceptionC(nn.Module):  # 定义InceptionC类

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)  # 第一轮卷积操作，输出通道数为192，卷积核大小为1*1

        c7 = channels_7x7  # 输出通道数为给定输出通道
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)  # 第二轮第一次卷积操作，卷积核为1*1
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        # 第二轮第二次卷积操作，卷积核大小为1*7，高度不填充，宽度填充3个像素
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        # 第二轮第三次卷积操作，输出通道数为192，卷积核大小为7*1，高度填充3个像素，宽度不填充
        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)  # 双倍卷积第一次，卷积核大小为1*1
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        # 双倍卷积第二次，卷积核大小为7*1，高度填充3个像素，宽度不填充
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        # 双倍卷积第三次，卷积核大小为1*7，高度不填充，宽度填充3个像素
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        # 双倍卷积第四次，卷积核大小为7*1，高度填充3个像素，宽度不填充
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        # 双倍卷积第五次，输出通道数为192，卷积核大小为1*7，高度不填充，宽度填充3个像素

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)  # 放入池化层内，输出通道数为192，卷积核大小为1*1

    def forward(self, x):  # 定义前向传播函数
        branch1x1 = self.branch1x1(x)  # 进行第一轮卷积

        branch7x7 = self.branch7x7_1(x)  # 进行第二轮卷积
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)  # 进行双倍卷积
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)  # 进行二位平均池化操作，池化核大小为3*3，步长为1，边缘填充1个像素
        branch_pool = self.branch_pool(branch_pool)  # 再次进行平均池化操作

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]  # 设置输出值
        return torch.cat(outputs, 1)  # 以branch7*7的维度将输出值进行拼接


class InceptionD(nn.Module):  # 定义InceptionD类

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)  # 第一层第一次卷积操作，输出通道数为192，卷积核大小为1*1
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)  # 第一层第二次卷积操作，输出通道数为320，卷积核大小为3*3，步长为2

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)  # 第二层第一次卷积操作，输出通道数为192，卷积核大小为1*1
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        # 第二层第二次卷积操作，输出通道数为192，卷积核大小为1*7，高度不填充，宽度填充3个像素
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        # 第二层第三次卷积操作，输出通道为192，卷积核大小为7*1，高度填充3个像素，宽度不填充
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)
        # 第二层第四次卷积操作，输出通道数为192，卷积核大小为3*3，步长为2

    def forward(self, x):  # 定义前向传播函数
        branch3x3 = self.branch3x3_1(x)  # 进行第一层卷积操作
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)  # 进行第二层卷积操作
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)  # 进行二位平均池化操作，池化核大小为3*3，步长为2
        outputs = [branch3x3, branch7x7x3, branch_pool]  # 设置输出值
        return torch.cat(outputs, 1)  # 以branch7*7*3的维度将输出值进行拼接


class InceptionE(nn.Module):  # 定义InceptionE类

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)  # 第一轮卷积，输出通道数为320，卷积核大小为1*1

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)  # 第二轮第一次卷积，输出通道为384，卷积核大小为1*1
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        # 第二轮第二次a卷积，输出通道为384，卷积核大小为1*3，高度不填充，宽度填充1个像素
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        # 第二轮第二次b卷积，输出通道为384，卷积核大小为3*1，高度填充1个像素，宽度不填充

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)  # 双倍卷积第一层，输出通道数为448，卷积核大小为1
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        # 双倍卷积第二层，输出通道为384，卷积核大小为3*3，边缘填充1个像素
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        # 双倍卷积第三层a，输出通道为384，卷积核大小为1*3，高度不填充，宽度填充1个像素
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        # 双倍卷积第三层b，输出通道为384，卷积核大小为3*1，高度填充1个像素，宽度不填充

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)  # 池化操作，输出通道数为192，池化核大小为1*1

    def forward(self, x):  # 定义前向传播函数
        branch1x1 = self.branch1x1(x)  # 进行第一轮卷积

        branch3x3 = self.branch3x3_1(x)  # 进行第二轮卷积
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)  # 以branch3*3的第一个维度的尺寸进行拼接

        branch3x3dbl = self.branch3x3dbl_1(x)  # 进行双倍卷积操作
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)  # 以branch3*3dbl的第一个维度的尺寸进行拼接

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)  # 二维平均池化操作，池化核为3*3，步长为1，边缘填充1个像素
        branch_pool = self.branch_pool(branch_pool)  # 进行二维平均池化操作

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]  # 设置输出值
        return torch.cat(outputs, 1)  # 以branch3*3的维度将输出值进行拼接


class InceptionAux(nn.Module):  # 定义InceptionAux类

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)  # 第一层卷积，输出通道数为128，卷积核为1*1
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)  # 第二层卷积，输出通道数为768，卷积核为5*5
        self.conv1.stddev = 0.01  # 设置标准偏差为0.01
        self.fc = nn.Linear(768, num_classes)  # 全连接网络
        self.fc.stddev = 0.001  # 设置全连接网络的标准偏差为1e-3

    def forward(self, x):  # 设置前向传播函数
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)  # 进行二维平均池化操作，池化核为5*5，步长为3
        # 5 x 5 x 768
        x = self.conv0(x)  # 进行第一层卷积
        # 5 x 5 x 128
        x = self.conv1(x)  # 进行第二层卷积
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)  # 将张量x转化为第一个维度的大小，并自动计算第二个维度的大小
        # 768
        x = self.fc(x)  # 进行全连接操作
        # 1000
        return x


class BasicConv2d(nn.Module):  # 定义BasicConv2d类

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)  # 图像卷积操作，不使用偏置操作
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)  # 将数据转换为标准正态分布，分母添加1e-3

    def forward(self, x):  # 定义前向传播函数
        x = self.conv(x)  # 进行图像卷积操作
        x = self.bn(x)  # 数据转化为标准正态分布
        return F.relu(x, inplace=True)  # 激活神经网络并且允许覆盖


if __name__ == '__main__':
    # 'Inception3'
    # Example
    net = Inception3()
    print(net)
