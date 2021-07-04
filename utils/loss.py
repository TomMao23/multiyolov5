# Loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general import bbox_iou
from utils.torch_utils import is_parallel
from torch.autograd import Variable


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))  # cls BCE yolov3后传统, BCE而不是CE
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))  # obj BCE

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets # 标签label smoothing后正样本和负样本的概率

        # Focal loss 是否focal loss以及γ的大小, 看超参数配置文件, 默认不用focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module # 若DDP外面又包了一层所有形式不一样
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box 上两行按推理的box放缩偏移
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target) 用CIoU loss
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes) 多类别才有cls loss
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']  # 三部分loss乘以其超参数配置文件中的增益
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()  # BCE自动平均故乘batchsize, detach()从计算图中除去防止梯度泄露

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


# 语义分割损失函数  带不带aux，带几个aux内外部接口本应该保持一致，但我没时间改了
class SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1, aux_num=2,
                 aux=False, aux_weight=0.1, weight=None,
                 ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)
        self.aux_num = aux_num

    def forward(self, *inputs):  # 这里接口写的很丑,没时间重构了,直接点就是无aux不用[],有aux几个结果输出用[]包装
        if not self.se_loss and not self.aux:  # 无aux, Base,PSP和Lab用这个
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:      
            if self.aux_num == 2:  # 两个aux，BiSe用这个
                pred1, pred2, pred3, target = tuple(inputs)
                loss1 = super(SegmentationLosses, self).forward(pred1, target)
                loss2 = super(SegmentationLosses, self).forward(pred2, target)
                loss3 = super(SegmentationLosses, self).forward(pred3, target)
                return loss1 + self.aux_weight*1.5 * loss2 + self.aux_weight/2.0 * loss3
            else:  # 一个aux, 目前没有用这个
                assert self.aux_num == 1
                pred1, pred2, target = tuple(inputs)
                loss1 = super(SegmentationLosses, self).forward(pred1, target)
                loss2 = super(SegmentationLosses, self).forward(pred2, target)
                return loss1 + self.aux_weight * loss2
        elif not self.aux:   # 目前仅支持以上三种配置(无aux，一个，两个)，以下两种情况目前未使用，所以bug未修改
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    # @staticmethod
    # def _get_batch_label_vector(target, nclass):
    #     # target is a 3D Variable BxHxW, output is 2D BxnClass
    #     batch = target.size(0)
    #     tvect = Variable(torch.zeros(batch, nclass))
    #     for i in range(batch):
    #         hist = torch.histc(target[i].cpu().data.float(),
    #                            bins=nclass, min=0,
    #                            max=nclass-1)
    #         vect = hist>0
    #         tvect[i] = vect
    #     return tvect


class SegFocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma=2, alpha=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='mean')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        if self.reduction == 'mean': return torch.mean(loss)
        elif self.reduction == 'sum': return torch.sum(loss)
        else: return loss


# 以下有两种OHEM实现略有不同，分别是根据两个bisenet实现修改的，aux使用接口一致, 第一种实现更快(batchsize＝12时候一轮比第二种快22s)
# bisenet的aux和main loss是同权重的[1.0, 1.0]但我的实验同权重非常不好，我认为辅助权重应该低于主权重，很多其他网络实现也是辅助权重低的
# 第一种
class OhemCELoss(nn.Module):  # 带ohem和aux的CE，0.7是根据bisenet原作者和复现者参数确定的
    def __init__(self, thresh=0.5, ignore_index=-1, aux=False, aux_weight=[0.15, 0.05]):  # 辅助损失可以设小，但bisenet里辅助损失系数为1(同权)，pytorch encoding项目默认0.2
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.aux = aux
        self.aux_weight = aux_weight

    def forward(self, preds, labels):
        if not self.aux:  # 此时preds应该为单个输出
            return self.forward_once(preds, labels)
        else:  # 此时preds应该为三个输出并用[]包裹起来，preds[0]永远是主输出
            mainloss = self.forward_once(preds[0], labels)  # 若采用resize标签到来节约显存参考下两行注释替换labels，并且去除yolo.py分割head的aux部分的上采样(不推荐resize标签)
            auxloss1 = self.forward_once(preds[1], labels)  #  (preds[1].shape[2], preds[1].shape[3]), mode='nearest')[0].long())
            auxloss2 = self.forward_once(preds[2], labels)  #  (preds[2].shape[2], preds[2].shape[3]), mode='nearest')[0].long())
            return mainloss + self.aux_weight[0] * auxloss1 + self.aux_weight[1] * auxloss2

    def forward_once(self, preds, labels):
        n_min = int(labels[labels != self.ignore_index].numel() // 16)  # 1/16=(1/4)^2即不计ignore的1/4张图  #(16*8**2)  # 最少样本公式是按bisenet原作者表达式写的(这个实现少除以8**2) 原式int(config.batch_size // len(engine.devices) * config.image_height * config.image_width //(16 * config.gt_down_sampling ** 2))
        # print(n_min)
        loss = self.criteria(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


# 第二种
class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index, reduction='mean', thresh=0.5, min_kept=256, aux=False, aux_weight=[0.4, 0.4], use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [1.4297, 1.4805, 1.4363, 3.365, 2.6635, 1.4311, 2.1943, 1.4817,
                 1.4513, 2.1984, 1.5295, 1.6892, 3.2224, 1.4727, 7.5978, 9.4117,
                 15.2588, 5.6818, 2.2067])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def forward(self, preds, target):
        if not self.aux:  # 此时preds应该为单个输出
            return forward_once(preds, target)
        else:  # 此时preds应该为三个输出并用[]包裹起来，preds[0]永远是主输出
            mainloss = self.forward_once(preds[0], target)  # 若采用resize标签到来节约显存参考下两行注释替换labels，并且去除yolo.py分割head的aux部分的上采样
            auxloss1 = self.forward_once(preds[1], target)  # F.interpolate(target.float().unsqueeze(0), (preds[1].shape[2], preds[1].shape[3]), mode='nearest')[0].long())
            auxloss2 = self.forward_once(preds[2], target)  # F.interpolate(target.float().unsqueeze(0), (preds[2].shape[2], preds[2].shape[3]), mode='nearest')[0].long())
            return mainloss + self.aux_weight[0] * auxloss1 + self.aux_weight[1] * auxloss2            

    def forward_once(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            logger.info('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)  # 注：pytorch1.2后bool tensor不支持１ - , use the `~` or `logical_not()` operator instead
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = torch.sort(mask_prob)
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(~valid_mask, self.ignore_index)  # 注：pytorch1.2后bool tensor不支持１ - , use the `~` or `logical_not()` operator instead
        target = target.view(b, h, w)

        return self.criterion(pred, target)