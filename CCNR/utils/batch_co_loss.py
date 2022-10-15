import torch
import torch.nn.functional as F

def memory_bank_change(memory_bank, rep_u, mask_class):
    for i in range(memory_bank.shape[0]):
        rep_u_i = rep_u[mask_class==i]
        add_num = rep_u_i.shape[0]
        if add_num == 0:
            continue
        if len(rep_u_i.shape) == 1:
            rep_u_i = rep_u_i.unsqueeze(0)
            add_num = 1
        rep_u_i = rep_u_i.unsqueeze(1)
        if add_num > 32:
            index = torch.randperm(rep_u_i.shape[0])[:32]
            rep_u_i = rep_u_i[index]
            add_num = 32
        memory_bank[i] = torch.cat((memory_bank[i][add_num:], rep_u_i), dim=0)

    return memory_bank

def memory_bank_change2(memory_bank2, rep, label, region=[16,16]):
    num = len(memory_bank2)
    dim = rep.shape[1]
    rep = F.interpolate(rep, region)
    label = F.interpolate(label.unsqueeze(1).float(), region)
    # label = label.long().repeat(1,dim,1,1)

    for i in range(num):
        b, _, x, y = torch.where(label == i)
        rep_i = rep[b, :, x, y]
        add_num = rep_i.shape[0]
        if add_num == 0:
            continue
        if len(rep_i.shape) == 1:
            rep_i = rep_i.unsqueeze(0)
            add_num = 1

        rep_i = rep_i.unsqueeze(1)
        # rep_i = rep_i.unsqueeze(0)
        memory_bank2[i][0] = torch.cat((memory_bank2[i][0,add_num:], rep_i), dim=0)


    return memory_bank2



def com_entr(img):
    img_unique = torch.unique(img, return_counts=True)[1]
    img_unique = img_unique / (4*4)
    img_unique = torch.sum(-img_unique * torch.log(img_unique))

    return img_unique



def entropy_sample(img, samplenum, b):
    assert img.shape[-1] == 64
    # img = img.permute(0, 2, 3, 1)
    img = torch.argmax(img, dim=0)

    entr_list = torch.zeros((256,3))
    # entr_sort =
    index = 0
    for i in range(16):
        for j in range(16):
            img_s = img[i*4:i*4+4, j*4:j*4+4]
            entr = com_entr(img_s)
            entr_list[index] = torch.tensor((i, j, entr))
            index += 1
    entr_list[:,2] = torch.softmax(entr_list[:,2] * 3, dim=0) # 3 can be changed===========
    query_dist = torch.distributions.categorical.Categorical(probs=entr_list[:,2])
    samp_class = query_dist.sample([samplenum])
    samp_u = torch.unique(samp_class, return_counts=True)
    entr_list[:, 2] = 0
    entr_list[samp_u[0], 2] = samp_u[1].float()
    #entr_list[5,:].unsqueeze(0).repeat_interleave(0,dim=0)
    xy = entr_list[:,:2].repeat_interleave(entr_list[:,-1].long(),dim=0)
    xy = xy * 4
    query_dist2 = torch.distributions.categorical.Categorical(probs=torch.tensor((0.25,0.25,0.25,0.25)))
    xy_r = query_dist2.sample([samplenum, 2])
    xy = xy + xy_r
    x = xy[:,0].long()
    y = xy[:,1].long()
    b = (torch.ones_like(x) * b).long()

    return b, x, y


def co_loss(pred, rep, label, memory_bank, query_num, memory_bank2, tau=0.2):
    with torch.no_grad():
        rep_ = rep
        label = label.long()
        triloss = 0
        memory_bank_mean = torch.mean(memory_bank, dim=1).squeeze(1)
        memory_bank_mean2 = memory_bank_mean.clone()
        for m in range(memory_bank_mean.shape[0]):
            memory_bank_mean2[m] = torch.mean(memory_bank2[m], dim=1).squeeze(1)
        rep = rep.permute(0, 2, 3, 1)
        b_all, x_all, y_all = torch.tensor([]), torch.tensor([]), torch.tensor([])
        for bb in range(rep.shape[0]):
            b, x, y = entropy_sample(pred[bb], query_num, bb)
            b_all = torch.cat((b_all,b))
            x_all = torch.cat((x_all, x))
            y_all = torch.cat((y_all, y))
        label_chos = label[b_all.long(), x_all.long(), y_all.long()]
        rep_chos = rep[b_all.long(), x_all.long(), y_all.long()]
        for i in range(memory_bank_mean.shape[0]):
            rep_i = rep_chos[label_chos == i]
            me_bank_oth = torch.cat((memory_bank_mean[0:i], memory_bank_mean[i + 1:]), dim=0)
            me_bank_i = memory_bank_mean[i]
            pos = torch.exp(torch.einsum('ij,j->i', rep_i, me_bank_i) / tau)
            neg = torch.sum(torch.exp(torch.einsum('ij,kj->ik', rep_i, me_bank_oth) / tau), dim=1)
            triloss_i = -torch.log(pos / (pos + neg))
            triloss_i = torch.sum(triloss_i)

            me_bank_oth2 = torch.cat((memory_bank_mean2[0:i], memory_bank_mean2[i + 1:]), dim=0)
            me_bank_i2 = memory_bank_mean2[i]
            pos2 = torch.exp(torch.einsum('ij,j->i', rep_i, me_bank_i2) / tau)
            neg2 = torch.sum(torch.exp(torch.einsum('ij,kj->ik', rep_i, me_bank_oth2) / tau), dim=1)
            triloss_i2 = -torch.log(pos2 / (pos2 + neg2))
            triloss_i2 = torch.sum(triloss_i2)

            triloss_isum = (triloss_i + triloss_i2) / 2
            triloss = 0.01*triloss_isum + triloss
        memory_bank = memory_bank_change(memory_bank, rep_chos, label_chos)
        memory_bank2 = memory_bank_change2(memory_bank2, rep_, label)

    return triloss, memory_bank




