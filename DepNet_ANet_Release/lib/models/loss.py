import torch
import torch.nn.functional as F
from IPython import embed

def bce_rescale_loss(scores, masks, sentence_masks, targets, cfg):
    # sentence_masks [b,set,1]
    sentence_masks = sentence_masks[:,:,0] #[b,set]
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    beta, gamma = cfg.BETA, cfg.GAMMA
    # joint_prob,scores,masks [b, sent, 1, 32, 32]
    joint_prob = torch.sigmoid(scores) * masks
    # joint_prob[0, 0, 0, :5,:5]
    # joint_prob[0, -1, 0, :5,:5]

    start_prob = joint_prob.max(-1).values
    start_prob = F.softmax(start_prob*beta, dim=-1)
    # [b,sent,1,32]
    #start_prob[0, 0, 0]

    N_clip = joint_prob.size(-1)
    start_time = torch.arange(0, N_clip).float()/float(N_clip) # [b,]
    start_time = start_time.repeat((start_prob.size(0), start_prob.size(1), 1, 1)).cuda()
    # [b,sent,1,32]
    #start_time[0, 0, 0]

    expect_start = start_prob * start_time # [b,sent,1,32]
    expect_start = expect_start.sum(-1) # [b,sent,1]
    # epect_start[0, :, 0], epect_start[1, :, 0]

    loss_order = 0.0
    tot_sent = 0
    for i in range(sentence_masks.size(0)):
        current_sentence_mask = sentence_masks[i]
        num_sent =  current_sentence_mask.sum().item()
        tot_sent += num_sent

        current_start = expect_start[i,:,0]-1

        diff = current_start[1:] - current_start[:-1]
        diff_mask = current_sentence_mask[1:]
        current_loss_instance = F.relu(-diff) * diff_mask

        loss_order += current_loss_instance.sum()
    loss_order = loss_order/tot_sent

    target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
    loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks
    loss_overlap = torch.sum(loss) / torch.sum(masks)

    loss_value = loss_overlap + gamma * loss_order
    return loss_value, loss_overlap, loss_order, joint_prob,