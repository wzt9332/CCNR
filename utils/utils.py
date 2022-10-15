import torch
import random
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import copy


def random_crop(data, label, logits,):
    type_index = []
    if data.dtype == torch.long :
        data = data.float()
        type_index.append('data')
    if label.dtype == torch.long :
        label = label.float()
        type_index.append('label')
    if logits.dtype == torch.long :
        logits = logits.float()
        type_index.append('logits')

    data_list, label_list, logits_list = [], [], []
    device = data.device
    data_dim = data.shape[3]
    crop_size = random.randint(int(data_dim*0.8), data_dim)
    crop_x = random.randint(0, data_dim - crop_size)
    crop_y = random.randint(0, data_dim - crop_size)
    data_shape = data.shape
    label_shape = label.shape
    logits_shape = logits.shape
    for k in range(data.shape[0]):
        if len(data_shape) == 4:
            data_new = data[k, :, crop_x:crop_x + crop_size, crop_y:crop_y + crop_size]
        elif len(data_shape) == 3:
            data_new = data[k, crop_x:crop_x + crop_size, crop_y:crop_y + crop_size]

        if len(label_shape) == 4:
            label_new = label[k, :, crop_x:crop_x + crop_size, crop_y:crop_y + crop_size]
        elif len(label_shape) == 3:
            label_new = label[k, crop_x:crop_x + crop_size, crop_y:crop_y + crop_size]

        if len(logits_shape) == 4:
            logits_new = logits[k, :, crop_x:crop_x + crop_size, crop_y:crop_y + crop_size]
        elif len(logits_shape) == 3:
            logits_new = logits[k, crop_x:crop_x + crop_size, crop_y:crop_y + crop_size]

        data_list.append(data_new.unsqueeze(0))
        label_list.append(label_new.unsqueeze(0))
        logits_list.append(logits_new.unsqueeze(0))

    data_trans, label_trans, logits_trans = torch.cat(data_list),\
        torch.cat(label_list), torch.cat(logits_list)

    if len(data_trans.shape) == 4:
        data_trans = F.interpolate(data_trans, [data_dim, data_dim], mode='nearest')
    elif len(data_trans.shape) == 3:
        data_trans = F.interpolate(data_trans.unsqueeze(1), [data_dim, data_dim], mode='nearest')
        data_trans = data_trans.squeeze(1)

    if len(label_trans.shape) == 4:
        label_trans = F.interpolate(label_trans, [data_dim, data_dim], mode='nearest')
    elif len(label_trans.shape) == 3:
        label_trans = F.interpolate(label_trans.unsqueeze(1), [data_dim, data_dim], mode='nearest')
        label_trans = label_trans.squeeze(1)

    if len(logits_trans.shape) == 4:
        logits_trans = F.interpolate(logits_trans, [data_dim, data_dim], mode='nearest')
    elif len(logits_trans.shape) == 3:
        logits_trans = F.interpolate(logits_trans.unsqueeze(1), [data_dim, data_dim], mode='nearest')
        logits_trans = logits_trans.squeeze(1)

    if 'data' in type_index:
        data_trans = data_trans.long()
    if 'label' in type_index:
        label_trans = label_trans.long()
    if 'logits' in type_index:
        logits_trans = logits_trans.long()


    return data_trans, label_trans, logits_trans






def generate_classmix_data(data, target, logits):
    batch_size, _, im_h, im_w = data.shape
    device = data.device

    new_data = []
    new_target = []
    new_logits = []
    for i in range(batch_size):
        mix_mask = generate_class_mask(target[i]).to(device)

        new_data.append((data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_target.append((target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

    new_data, new_target, new_logits = torch.cat(new_data), torch.cat(new_target), torch.cat(new_logits)
    return new_data, new_target.long(), new_logits



def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)  # all unique labels
    labels_select = labels[torch.randperm(len(labels))][:len(labels) // 2]  # randomly select half of labels

    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()


# --------------------------------------------------------------------------------
# Define Polynomial Decay
# --------------------------------------------------------------------------------
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]






# --------------------------------------------------------------------------------
# Define training losses
# --------------------------------------------------------------------------------
def compute_supervised_loss(predict, target, reduction=True):
    if reduction:
        loss = F.cross_entropy(predict, target, ignore_index=-1)
    else:
        loss = F.cross_entropy(predict, target, ignore_index=-1, reduction='none')
    return loss



def compute_unsupervised_loss(predict, target, logits, strong_threshold=0.7):
    batch_size = predict.shape[0]
    valid_mask = (target >= 0).float()   # only count valid pixels

    weighting = logits.view(batch_size, -1).ge(strong_threshold).sum(-1) / valid_mask.view(batch_size, -1).sum(-1)
    loss = F.cross_entropy(predict, target, reduction='none', ignore_index=-1)
    weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None] * loss, loss > 0))
    return weighted_loss



def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
    # we will still mask out those invalid values in valid mask
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)



def query_prob(negative_list):
    negative_mean = torch.mean(negative_list, dim=1)
    out_prob = []
    for i in range(negative_mean.shape[0]):
        n = negative_mean[i]
        # cos = torch.softmax(torch.sum(torch.pairwise_distance(n.unsqueeze(0), negative_mean), dim=1), dim=0)
        cos = torch.exp(torch.cosine_similarity(n, negative_mean.squeeze(1), dim=1))
        cos[i] = 0
        out_prob.append(torch.sum(cos))

    out_prob = torch.softmax(torch.tensor(out_prob), dim=0)
    return out_prob




# --------------------------------------------------------------------------------
# Define evaluation metrics
# --------------------------------------------------------------------------------
class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item(), acc.item()




