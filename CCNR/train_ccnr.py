import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import argparse
import time
from utils.dataset import *
from utils.utils import *
import logging
from networks.ccnr_net import CCNR
from utils.batch_co_loss import co_loss
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


device = torch.device('cuda')

def parse_args():
    parser = argparse.ArgumentParser(description='Trainning of wzt')
    parser.add_argument('--data_dir', type=str, default='./data', help='The root dir of data.')
    parser.add_argument('--data_set', type=str, default='hongkong', help='The dir name of dataset in data root_dir.')
    parser.add_argument('--classes_num', type=int, default=9, help='The number of classes.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size.')
    parser.add_argument('--epoch', type=int, default=2500, help='The epoch of training.')
    parser.add_argument('--log_file', type=str, default='./log/log_supvised7.txt', help='log file.')
    parser.add_argument('--save_path', type=str, default='./model', help='network save path')
    parser.add_argument('--lr', default=0.002, help='learning rate')
    parser.add_argument('--memory_bank', type=int, default=1000, help='negative point lists')
    parser.add_argument('--r_class_num', type=list, default=[8993, 174169, 45594, 6156,
                                                             17605, 16741, 2929, 5905, 9140],
                        help='train data number, initialization memory_bank2')
    parser.add_argument('--out_size', type=int, default=4096, help='the size of output')
    parser.add_argument('--rep_dim', type=int, default=256, help='dim of representation')
    parser.add_argument('--unsup_threshold', type=float, default=0.7, help='unlabeled data threshold')
    parser.add_argument('--sup_threshold', type=float, default=0.97, help='labeled data threshold')
    parser.add_argument('--query_num', type=int, default=768, help='the number of query pixels in a image')
    parser.add_argument('--negative_num', type=int, default=256, help='the number of negative pixels about a positive pixel')
    #  ----------------------------------------
    parser.add_argument('--num_workers', type=int, default=4, help='dataLoader worker number.')
    parser.add_argument('--num_threads', type=int, default=8, help='cpu threads worker number.')
    parser.add_argument('--in_ch', type=int, default=1, help='The number of input channel.')
    parser.add_argument('--save_epoch', type=int, default=100, help='network save interval')
    parser.add_argument('--batch_print', type=int, default=40, help='Print loss in interval')


    return parser



#==============
def fish2(x):
    out = 4 * np.exp(2*x) * np.exp(-2 * np.exp(x))
    return out
n_step = 0.0001
n_x = np.arange(-10,10,n_step)
n_y = fish2(n_x)
nosie_make = Categorical(torch.tensor(n_y))
def noise_mean(x, step):
    out = (x - 10 / step) * step - (-0.577215664901532860 + 1 - np.log(2))
    return out/255

def add_noise_to_img(img, nosie_make, n_step, device):
    n0 = nosie_make.sample(img.shape)
    n0 = n0.to(device)
    n0 = noise_mean(n0, n_step)
    img[img == 0] = 10e-8
    img = torch.log(img)
    img = img - n0
    img = torch.exp(img)
    return img


# ========================


def main():
    arg = parse_args()
    cfg = arg.parse_args()
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(cfg.log_file)
    console = logging.StreamHandler()
    logger.addHandler(handler)
    logger.addHandler(console)
    logger.info('-----------Trainning Start----------')
    logger.info(time.asctime(time.localtime(time.time())))
    logger.info('class number: {}'.format(cfg.classes_num))
    # data loader ----------------------------
    torch.set_num_threads(cfg.num_threads)

    #========= dataloader===========
    data_loader = BuildDataLoader(os.path.join(cfg.data_dir, cfg.data_set), cfg.batch_size, cfg.classes_num)
    train_l_loader, test_loader = data_loader.build(supervised=False)

    train_epoch = 200
    test_epoch = len(test_loader)
    avg_cost = np.zeros((cfg.epoch, 3))

    #==========model and optimizer===========
    model1 = CCNR(models.resnet101(pretrained=True), num_classes=cfg.classes_num, output_dim=256)
    model1 = torch.nn.DataParallel(model1).cuda()
    optimizer1 = optim.SGD(model1.parameters(), lr=cfg.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
    scheduler1 = PolyLR(optimizer1, cfg.epoch, power=0.9)


    MSE_LOSE = torch.nn.MSELoss()

    #==========memory bank=============
    memory_bank = torch.Tensor(cfg.classes_num, cfg.memory_bank, 1, cfg.rep_dim).uniform_(-1,1).cuda()
    memory_bank = nn.functional.normalize(memory_bank, dim=3)

    memory_bank2 = []
    for c in cfg.r_class_num:
        memory_b2 = torch.Tensor(1, c, 1, cfg.rep_dim).uniform_(-1, 1).cuda()
        memory_b2 = nn.functional.normalize(memory_b2, dim=3)
        memory_bank2.append(memory_b2)

    for index in range(cfg.epoch):
        train_l_dataset = iter(train_l_loader)

        model1.eval()

        for i in range(train_epoch):
            train_l_data, train_l_label = train_l_dataset.next()
            train_l_data, train_l_label = train_l_data.cuda(), train_l_label.cuda()

            optimizer1.zero_grad()

            # model1 training=================
            model1.train()
            if i % 2 == 0:
                for name, p in model1.named_parameters():
                    if name.startswith('classifier') or name.startswith('representation'):
                        p.requires_grad = True
                    if name.startswith('restore_layers'):
                        p.requires_grad = False

                pred_l1, rep, reimg = model1(train_l_data)
                pred_l1_large = F.interpolate(pred_l1, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)

                # supervised-learning loss
                sup_loss = compute_supervised_loss(pred_l1_large, train_l_label)
                # co loss======
                triloss, memory_bank = co_loss(pred_l1, rep, train_l_label, memory_bank, cfg.query_num,
                                               memory_bank2, tau=0.2)
                # reimg loss and noise loss
                re_img_loss = MSE_LOSE(reimg, train_l_data)


                loss = sup_loss + triloss + re_img_loss
                loss.backward()
                optimizer1.step()
                logger.info('{}, {}, {}, {}, {}, {}'.format(i, loss, sup_loss, triloss, re_img_loss, scheduler1.get_lr()[0]))
            else:
                # ========== add zero-mean noise
                train_l_data0 = add_noise_to_img(train_l_data, nosie_make, n_step, device)
                train_l_data1 = add_noise_to_img(train_l_data, nosie_make, n_step, device)
                # ======================================
                for name, p in model1.named_parameters():
                    if name.startswith('classifier') or name.startswith('representation'):
                        p.requires_grad = False
                    if name.startswith('restore_layers'):
                        p.requires_grad = True
                _, _, reimg = model1(train_l_data0)

                # reimg loss and noise loss
                re_img_loss = MSE_LOSE(reimg, train_l_data1)

                loss = re_img_loss
                loss.backward()
                optimizer1.step()
                logger.info(
                    '{}, {}, {}'.format(i, loss, re_img_loss))

        scheduler1.step()



        # ============= test ================
        with torch.no_grad():
            model1.eval()
            test_dataset = iter(test_loader)
            conf_mat = ConfMatrix(data_loader.num_segments)
            for i in range(test_epoch):
                test_data, test_label = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.to(device)
                test_label = test_label.long()

                pred, _, _r = model1(test_data)
                pred = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)
                loss = compute_supervised_loss(pred, test_label)
                conf_mat.update(pred.argmax(1).flatten(), test_label.flatten())
            avg_cost[index, 0] = loss
            avg_cost[index, 1:] = conf_mat.get_metrics()

        logger.info('epoch{}: ,loss:{} , mIoU:{} , Acc:{}'.format(index,avg_cost[index,0],avg_cost[index,1],
                                                                  avg_cost[index,2]))
        logger.info('TOP: {}, {}'.format(avg_cost[:, 1].max(), avg_cost[:, 2].max()))
        if avg_cost[index][1] >= avg_cost[:, 1].max():
            checkpoint = {
                'model': model1.state_dict(),
            }
            torch.save(checkpoint, cfg.save_path + '/ccnr.pth')


if __name__ == '__main__':
    main()
