from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


class LoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace

        self.loss_config = config['loftr']['loss']
        self.match_type = 'dual_softmax'
        self.sparse_spvs = self.config['loftr']['match_coarse']['sparse_spvs']
        self.fine_sparse_spvs = self.config['loftr']['match_fine']['sparse_spvs']
        
        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        # coarse_overlap_weight
        self.overlap_weightc = self.config['loftr']['loss']['coarse_overlap_weight']
        self.overlap_weightf = self.config['loftr']['loss']['fine_overlap_weight']
        # subpixel-level
        self.local_regressw = self.config['loftr']['fine_window_size']
        self.local_regress_temperature = self.config['loftr']['match_fine']['local_regress_temperature']
        

    def compute_coarse_loss(self, conf, conf_gt, weight=None, overlap_weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        del conf_gt
        # logger.info(f'real sum of conf_matrix_c_gt: {pos_mask.sum().item()}')
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']
            
            if self.sparse_spvs:
                pos_conf = conf[pos_mask]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                if self.overlap_weightc:
                    loss_pos = loss_pos * overlap_weight # already been masked slice in supervision

                loss = c_pos_w * loss_pos.mean()
                return loss
            else:  # dense supervision
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                logger.info("conf_pos_c: {loss_pos}, conf_neg_c: {loss_neg}".format(loss_pos=conf[pos_mask].mean(), loss_neg=conf[neg_mask].mean()))
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                if self.overlap_weightc:
                    loss_pos = loss_pos * overlap_weight # already been masked slice in supervision

                loss_pos_mean, loss_neg_mean = loss_pos.mean(), loss_neg.mean()
                logger.info("conf_pos_c: {loss_pos}, conf_neg_c: {loss_neg}".format(loss_pos=conf[pos_mask].mean(), loss_neg=conf[neg_mask].mean()))
                return c_pos_w * loss_pos_mean + c_neg_w * loss_neg_mean
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))

    def compute_fine_loss(self, conf_matrix_f, conf_matrix_f_gt, overlap_weight=None):
        """
        Args:
            conf_matrix_f (torch.Tensor): [m, WW, WW] <x, y>
            conf_matrix_f_gt (torch.Tensor): [m, WW, WW] <x, y>
        """
        if conf_matrix_f_gt.shape[0] == 0:
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                            # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                pass
            else:
                return None
        pos_mask, neg_mask = conf_matrix_f_gt == 1, conf_matrix_f_gt == 0
        del conf_matrix_f_gt
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            c_neg_w = 0.

        conf = torch.clamp(conf_matrix_f, 1e-6, 1-1e-6)
        alpha = self.loss_config['focal_alpha']
        gamma = self.loss_config['focal_gamma']
        
        if self.fine_sparse_spvs:
            loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
            if self.overlap_weightf:
                loss_pos = loss_pos * overlap_weight # already been masked slice in supervision
            return c_pos_w * loss_pos.mean()
        else:
            loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
            loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
            logger.info("conf_pos_f: {loss_pos}, conf_neg_f: {loss_neg}".format(loss_pos=conf[pos_mask].mean(), loss_neg=conf[neg_mask].mean()))
            if self.overlap_weightf:
                loss_pos = loss_pos * overlap_weight # already been masked slice in supervision

            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()

    
    def _compute_local_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()
    
    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None])
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        if self.overlap_weightc:
            loss_c = self.compute_coarse_loss(
                data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn' \
                    else data['conf_matrix'],
                data['conf_matrix_gt'],
                weight=c_weight, overlap_weight=data['conf_matrix_error_gt'])
        
        else:
            loss_c = self.compute_coarse_loss(
                data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn' \
                    else data['conf_matrix'],
                data['conf_matrix_gt'],
                weight=c_weight)

        loss = loss_c * self.loss_config['coarse_weight']
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})
        
        # 2. pixel-level loss (first-stage refinement)
        if self.overlap_weightf:
            loss_f = self.compute_fine_loss(data['conf_matrix_f'], data['conf_matrix_f_gt'], data['conf_matrix_f_error_gt'])
        else:
            loss_f = self.compute_fine_loss(data['conf_matrix_f'], data['conf_matrix_f_gt'])
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f":  loss_f.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound

        # 3. subpixel-level loss (second-stage refinement)
        # we calculate subpixel-level loss for all pixel-level gt
        if 'expec_f' not in data:
            sim_matrix_f, m_ids, i_ids, j_ids_di, j_ids_dj = data['sim_matrix_ff'], data['m_ids_f'], data['i_ids_f'], data['j_ids_f_di'], data['j_ids_f_dj']
            del data['sim_matrix_ff'], data['m_ids_f'], data['i_ids_f'], data['j_ids_f_di'], data['j_ids_f_dj']
            delta = create_meshgrid(3, 3, True, sim_matrix_f.device).to(torch.long) # [1, 3, 3, 2]
            m_ids = m_ids[...,None,None].expand(-1, 3, 3)
            i_ids = i_ids[...,None,None].expand(-1, 3, 3)
            # Note that j_ids_di & j_ids_dj in (i, j) format while delta in (x, y) format
            j_ids_di = j_ids_di[...,None,None].expand(-1, 3, 3) + delta[None, ..., 1]
            j_ids_dj = j_ids_dj[...,None,None].expand(-1, 3, 3) + delta[None, ..., 0]

            sim_matrix_f = sim_matrix_f.reshape(-1, self.local_regressw*self.local_regressw, self.local_regressw+2, self.local_regressw+2) # [M, WW, W+2, W+2]
            sim_matrix_f = sim_matrix_f[m_ids, i_ids, j_ids_di, j_ids_dj]
            sim_matrix_f = sim_matrix_f.reshape(-1, 9)

            sim_matrix_f = F.softmax(sim_matrix_f / self.local_regress_temperature, dim=-1)
            heatmap = sim_matrix_f.reshape(-1, 3, 3)
            
            # compute coordinates from heatmap
            coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]
            data.update({'expec_f': coords_normalized})
        loss_l = self._compute_local_loss_l2(data['expec_f'], data['expec_f_gt'])

        loss += loss_l * self.loss_config['local_weight']
        loss_scalars.update({"loss_l":  loss_l.clone().detach().cpu()})

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})