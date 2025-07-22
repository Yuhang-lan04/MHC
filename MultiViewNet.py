import torch  # 导入PyTorch库
from torch import nn  # 从torch中导入神经网络模块
from torch.nn import functional as F  # 从torch.nn中导入功能模块

class MultiViewNet(nn.Module):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=3):
        super(MultiViewNet, self).__init__()
        self.module_name = "multi_view_model"

        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
        self.fc = nn.Sequential(*modules)
        self.norm = norm

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            if out.dim() == 1:
                norm_x = torch.norm(out, dim=0, keepdim=True)
            else:
                norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out
