import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights


def criterion(preds, targets, target_lengths):
    bs, _ = targets.size()
    log_probs = F.log_softmax(preds, 2)
    input_lengths = torch.full(
        size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
    )
    # print(f'----\n{preds.size()}\n{target_lengths}\n{input_lengths}')
    loss = nn.CTCLoss(blank=0)(
        log_probs, targets, input_lengths, target_lengths
    )
    return loss

class ResNetRNNModel(nn.Module):
    def __init__(self, num_l_chars=13, num_m_chars=55, num_u_chars=20):
        super(ResNetRNNModel, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-5]))
        self.linear_1 = nn.Linear(2048, 64)
        self.drop_1 = nn.Dropout(0.2)
        self.lstm = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output_1 = nn.Linear(64, num_l_chars)
        self.output_2 = nn.Linear(64, num_m_chars)
        self.output_3 = nn.Linear(64, num_u_chars)

    def forward(self, images, targets_1=None, targets_2=None, targets_3=None, lengths=None):
        bs, _, _, _ = images.size()
        x = self.resnet(images)
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)
        x, _ = self.lstm(x)
        x1 = self.output_1(x)
        x1 = x1.permute(1, 0, 2)
        x2 = self.output_2(x)
        x2 = x2.permute(1, 0, 2)
        x3 = self.output_3(x)
        x3 = x3.permute(1, 0, 2)

        if targets_1 is not None:
            loss1 = criterion(x1, targets_1, lengths)
            loss2 = criterion(x2, targets_2, lengths)
            loss3 = criterion(x3, targets_3, lengths)
            if loss1 < 0:
                print(f'\n{loss1}, {targets_1}')
            return (x1, x2, x3), loss1 + loss2 + loss3

        return  (x1, x2, x3), None
