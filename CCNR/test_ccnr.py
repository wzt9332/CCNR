import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
from utils.dataset import *
from utils.utils import *
from networks.ccnr_net import CCNR
import torchvision.models as models
import numpy as np
import cv2


device = torch.device('cuda')



def parse_args():
    parser = argparse.ArgumentParser(description='Trainning of wzt')
    parser.add_argument('--data_dir', type=str, default='./data', help='The root dir of data.')
    parser.add_argument('--data_set', type=str, default='hongkong', help='The dir name of dataset in data root_dir.')
    parser.add_argument('--classes_num', type=int, default=9, help='The number of classes.')
    parser.add_argument('--weight_path', type=str, default='./model/ccnr.pth', help='The weight path')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size.')
    parser.add_argument('--epoch', type=int, default=2500, help='The epoch of training.')
    parser.add_argument('--save_path', type=str, default='./model', help='network save path')
    parser.add_argument('--lr', default=0.002, help='learning rate')
    parser.add_argument('--memory_bank', type=int, default=256, help='negative point lists')
    parser.add_argument('--rep_dim', type=int, default=256, help='dim of representation')
    parser.add_argument('--unsup_threshold', type=float, default=0.7, help='unlabeled data threshold')
    parser.add_argument('--sup_threshold', type=float, default=0.97, help='labeled data threshold')
    parser.add_argument('--query_num', type=int, default=256, help='the number of query pixels in a image')
    parser.add_argument('--negative_num', type=int, default=256, help='the number of negative pixels about a positive pixel')
    #  ----------------------------------------
    parser.add_argument('--num_workers', type=int, default=4, help='dataLoader worker number.')
    parser.add_argument('--num_threads', type=int, default=8, help='cpu threads worker number.')
    parser.add_argument('--in_ch', type=int, default=1, help='The number of input channel.')
    parser.add_argument('--save_epoch', type=int, default=100, help='network save interval')
    parser.add_argument('--batch_print', type=int, default=40, help='Print loss in interval')


    return parser




color_map = [[0,0,0],[255,0,0],[0,255,0],[0,255,255],
             [255,0,255],[20,80,130],[0,0,255],
             [255,255,0],[140,180,210]] #hk

# color_map = [[0,0,0],[255,0,0],[0,255,0],[0,0,255],
#              [140,180,210]] #ist

color_map = [[0,0,0],[255,0,0],[0,255,0],[0,255,255],
             [0,0,255],[140,180,210],[255,255,0]]

def color_drwa(label, color_map):
    img = np.zeros((label.shape[0], label.shape[1], 3))
    for i in range(len(color_map)):
        x, y = np.where(label == i)
        img[x, y] = color_map[i]
    return img


def main():
    arg = parse_args()
    cfg = arg.parse_args()

    # data loader ----------------------------
    torch.set_num_threads(cfg.num_threads)

    #========= dataloader===========
    data_loader = BuildDataLoader(os.path.join(cfg.data_dir, cfg.data_set), cfg.batch_size, cfg.classes_num)
    train_l_loader, test_loader = data_loader.build(supervised=False)


    test_epoch = len(test_loader)
    avg_cost = np.zeros((10, 3))

    #==========model and optimizer===========
    model1 = CCNR(models.resnet101(pretrained=True), num_classes=cfg.classes_num, output_dim=256)
    weight = torch.load(cfg.weight_path)['model']

    model1 = torch.nn.DataParallel(model1).cuda()
    model1.load_state_dict(weight)

    # ============= test ================
    namelist = test_loader.dataset.idx_list
    index = 0
    with torch.no_grad():
        model1.eval()
        test_dataset = iter(test_loader)
        conf_mat = ConfMatrix(data_loader.num_segments)
        for i in range(test_epoch):
            test_data, test_label = test_dataset.next()
            test_data, test_label = test_data.to(device), test_label.to(device)
            test_label = test_label.long()

            pred, rep, _ = model1(test_data)

            pred = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)
            loss = compute_supervised_loss(pred, test_label)
            conf_mat.update(pred.argmax(1).flatten(), test_label.flatten())

            pred = torch.argmax(pred, dim=1)
            for j in range(pred.shape[0]):
                img = pred[j].cpu().data.numpy()
                img = color_drwa(img, color_map)
                cv2.imwrite('./output/' + namelist[index] + '.jpg', img)
                index = index + 1




        avg_cost[0, 0] = loss
        avg_cost[0, 1:] = conf_mat.get_metrics()

    print('test: ,loss:{} , mIoU:{} , Acc:{}'.format(avg_cost[0, 0], avg_cost[0, 1],
                                                     avg_cost[0, 2]))




if __name__ == '__main__':
    main()
