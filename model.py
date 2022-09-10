import torch
from torch import nn
from torch.nn import functional as F


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


class SinhalaOCRModel(nn.Module):
    def __init__(self, num_l_chars=13, num_m_chars=55, num_u_chars=20):
        super(SinhalaOCRModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 6), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear_1 = nn.Linear(1152, 64)
        self.drop_1 = nn.Dropout(0.2)
        self.lstm = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output_1 = nn.Linear(64, num_l_chars)
        self.output_2 = nn.Linear(64, num_m_chars)
        self.output_3 = nn.Linear(64, num_u_chars)
        self.temp = True

    def forward(self, images, targets_1=None, targets_2=None, targets_3=None, lengths=None):
        bs, _, _, _ = images.size()
        x = F.relu(self.conv_1(images))
        x = self.pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)
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


"""if __name__ == "__main__":
    cm = SinhalaOCRModel(19)
    img = torch.rand((1, 3, 50, 200))
    x, _ = cm(img, torch.rand((1, 5)))"""
