import copy

from torch.optim import lr_scheduler
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from training.dataset import *
from common import *

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

bpt_name = 'breakpoint-pars.pkl'


def check_accuracy(loader,
                   model,
                   iter_n,
                   writer,
                   device=torch.device('cpu'),
                   dtype=torch.uint8,
                   ):
    """
    :param loader: validation data or test data loader
    :param model: trained model
    :param iter_n: current iteration number (for logging)
    :param test_acc_log: log the test accuracy in this array
    :param device: cpu or cuda
    :param dtype: data type
    :return: None
    """
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        loader_iter = iter(loader)
        sample_batched = loader_iter.next()
        x = sample_batched['img']
        y = sample_batched['corner']
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=dtype)
        corners = model(x)
        dist = ((corners - y) ** 2).mean()
        writer.add_scalar('validation loss', dist, iter_n)
        print('validation dist={}'.format(dist))
    return dist  # only one batch


def train_(model,
           optimizer,
           loader_train,
           loader_val,
           writer,
           print_every=10,
           epochs=1,
           device=torch.device('cpu'),
           scheduler=None
           ):
    """
    :param model: pytorch model to be trained
    :param optimizer: pytorch optimizer
    :param loader_train:
    :param loader_val:
    :param writer: SummaryWriter
    :param print_every: check test_accuracy every print_every iterations
    :param epochs: the number of traverse on the whole training set
    :param device: cpu or cuda
    :param dtype: data type
    :return: best_model
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    lowest_dist = None
    best_model = None
    count = 0
    for e in range(epochs):
        for t, sample_batched in enumerate(loader_train):
            model.train()  # put model to training mode
            x = sample_batched['img']
            y = sample_batched['corner']
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)

            corners = model(x)
            loss = ((corners - y) ** 2).mean()
            writer.add_scalar('training loss', loss, count)
            writer.add_scalar('shift', corners[0][0].item(), count)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Epoch %d, Iteration %d, Total %d, loss = %.4f' % (e, t, count, loss.item()))
                val_dist = check_accuracy(loader=loader_val,
                                          model=model,
                                          iter_n=count,
                                          writer=writer,
                                          device=device,
                                          dtype=dtype, )
                print()
                if not lowest_dist or val_dist < lowest_dist:
                    best_model = copy.deepcopy(model)
                    lowest_dist = val_dist
            count += 1
            if scheduler:
                scheduler.step()
        model_dir = os.path.join(cache_dir, 'train' + str(train_num))
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_path = os.path.join(model_dir, 'model_breakpoint.pth')
        print('model breakpoint saved to:', model_path)
        torch.save(model, model_path)
        with open(os.path.join(model_dir, bpt_name), 'wb') as f:
            pickle.dump(epoch, f)
    print('lowest dist:', lowest_dist)
    return best_model, lowest_dist


def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Net(nn.Module):

    def __init__(self, win_size_actual):
        super(Net, self).__init__()
        channel_in = 1
        channel_1 = 5
        kernel_size_1 = (3, 3)
        self.conv1 = nn.Conv2d(channel_in, channel_1, kernel_size_1, padding=1)
        self.fc1 = nn.Linear((win_size_actual ** 2) * channel_1, 2, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = flatten(x)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    USE_GPU = True
    dtype = torch.float32
    batch_size = 640
    win_size = 5
    epoch = 2
    learning_rate = 5e-6
    win_size_actual = win_size * 2 + 1
    lr_decay = 1
    # storing directory settings  ______________________________________
    train_num = 0
    mid_dir_name = 'train'
    while os.path.exists(os.path.join(log_dir, mid_dir_name + str(train_num))):
        train_num += 1

    log_dir = os.path.join(log_dir, mid_dir_name + str(train_num))
    parameter_file_path = os.path.join(log_dir, 'parameter.txt')

    model_dir = os.path.join(cache_dir, mid_dir_name + str(train_num))
    model_path = os.path.join(model_dir, 'model.pth')
    model_breakpoint_path = os.path.join(model_dir, 'model_breakpoint.pth')

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    with open(parameter_file_path, 'w+') as f:
        f.write('batch_size:\t{}\n'.format(batch_size))
        f.write('learning_rate:\t{}\n'.format(learning_rate))
        f.write('epoch:\t{}\n'.format(epoch))
        f.write('win_size:\t{}\n'.format(win_size))
        f.write('lr_decay:\t{}\n'.format(lr_decay))

    writer = SummaryWriter(log_dir)
    # dataset settings _______________________________________________________
    test_dir = os.path.join(data_dir, 'test')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    train_dataset = SubpixelDataset(train_dir, win_size=win_size)
    val_dataset = SubpixelDataset(val_dir, win_size=win_size)
    loader_train = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler.SubsetRandomSampler(range(len(train_dataset))))
    loader_val = DataLoader(val_dataset, batch_size=10000, shuffle=False)
    # GPU ______________________________________________________________________________
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using device:', device)
    # # pretrained model __________________________________________________________________
    # model_conv = models.resnet18(pretrained=True)
    # num_ftrs = model_conv.fc.in_features
    # model_conv.fc = nn.Linear(num_ftrs, 2)
    # model_conv = model_conv.to(device)
    # optimizer_conv = optim.SGD(model_conv.parameters(),
    #                            lr=learning_rate, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=500, gamma=lr_decay)
    model = Net(win_size_actual)
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=lr_decay)
    # training ___________________________________________________________________________
    model, lowest_dist = train_(model=model,
                                     optimizer=optimizer,
                                     loader_train=loader_train,
                                     loader_val=loader_val,
                                     writer=writer,
                                     device=device,
                                     epochs=epoch,
                                     print_every=30,
                                     scheduler=exp_lr_scheduler)
    # use the best model

    # logging and saving __________________________________________________-
    model.eval()
    model = model.to(device=torch.device('cpu'), dtype=dtype)
    torch.save(model, model_path)
    print('model saved to:', model_path)
    win_size_path = os.path.join(model_dir, 'win_size.pkl')
    with open(win_size_path, 'wb') as f:
        pickle.dump(win_size, f)
        print('win_size saved to:', win_size_path)

    with open(parameter_file_path, 'a') as f:
        f.write('lowest_dist:\t{}\n'.format(lowest_dist))

    writer.close()
