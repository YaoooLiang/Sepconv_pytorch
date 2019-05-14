from model import Net
import torch
from dataloader import vimeo_dataloader


class L1_Charbonnier_loss(torch.nn.Module):
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


def main():

    train_loader = torch.utils.data.DataLoader(vimeo_dataloader("/root/toflow_keras/dataset/vimeo_triplet"),
                                                batch_size=8,
                                                shuffle=True,
                                                num_workers=8,
                                                pin_memory=True,
                                                drop_last=True)
    model = Net()
    # model._check_gradients(SeparableConvolutionModule())
    model = model.cuda()
    criterion = L1_Charbonnier_loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(50):
        train(train_loader, model, optimizer, criterion, epoch)
        check_point_filename = 'weights/checkpoint_epoch_%02d.pth' % epoch
        torch.save(model.state_dict(), check_point_filename)


def train(train_loader, model, optimizer, criterion, epoch):
    losses = AverageMeter()

    model.train()

    for i, (input1, target, input2) in enumerate(train_loader):
        # print(input1.size())
        # inputs = inputs.permute(1, 0, 2, 3, 4)
        # target = target.cuda(async=True)
        input1_var = torch.autograd.Variable(input1).cuda()
        input2_var = torch.autograd.Variable(input2).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        predict = model(input1_var, input2_var)

        loss1 = criterion(predict, target_var)

        # loss2 = criterion(output2, target)

        loss = loss1

        losses.update(loss.item(), input1.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            # check_point_filename = 'checkpoint_epoch_%02d.pth' % i
            # torch.save(model.state_dict(), check_point_filename)

            print('epoch: {} \t step: {} \t loss: {}'.format(epoch, i, losses.avg))
            losses.reset()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
