import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.operator import Operator
from utils.types import M_TRAIN, BaseType, frac_open_0
from utils.math_ops import harmony, is_bin_times
from data.penn_types import C_ABSTRACT
from time import time
from math import log
from models.utils import PCA, fraction, hinge_score
from models.loss import binary_cross_entropy, hinge_loss, big_endian_height_mask
from experiments.helper import warm_adam
from experiments.t_lstm_tokenize.types import D_NOISE, D_CLEAN

train_type = dict(learning_rate = BaseType(0.001, validator = frac_open_0))

class TokenizerOperator(Operator):
    def __init__(self, model, get_datasets, recorder, i2vs, train_config):
        super().__init__(model, get_datasets, recorder, i2vs)
        self._sigmoid = nn.Sigmoid()
        self._validity_hinge_loss = False
        self._vis = {}
        self._train_config = train_config

    def _build_optimizer(self, start_epoch):
        self._loss_weights = 0.49, 0.01, 0.5
        self._writer = SummaryWriter(self.recorder.create_join('train'))
        optim, schedule_lr = warm_adam(self._model, self._train_config.learning_rate)
        self._schedule_lr = schedule_lr
        return optim

    def _schedule(self, epoch, wander_ratio):
        learning_rate = self._schedule_lr(epoch, wander_ratio)
        self._writer.add_scalar('Batch/Learning_Rate', learning_rate, self.global_step)
        self._writer.add_scalar('Batch/Epoch', epoch, self.global_step)

    def _step(self, mode, ds_name, batch, flush = True, batch_id = None):
        # assert ds_name == C_ABSTRACT
        if mode == M_TRAIN and flush:
            self.optimizer.zero_grad()

        batch_time = time()
        token_ids, offset, length = (batch[key] for key in ('token', 'offset', 'length'))
        if ds_name == D_NOISE:
            (batch_size, batch_len, static, dynamic,
             existence, validity_logit) = self._model(token_ids, None, None, noise_mode = True)
            batch_time = time() - batch_time

            validity_logit.squeeze_(dim = 2)
            if self._validity_hinge_loss:
                validity = validity_logit > 0
            else:
                validity_logit = self._sigmoid(validity_logit)
                validity = validity_logit > 0.5

            valid = batch['first_validity'], batch['second_validity']
            valid = torch.cat(valid, dim = 1)
            existence.squeeze_(dim = 2)
            validity_match = (validity == valid) & existence

            if mode == M_TRAIN:
                if self._validity_hinge_loss:
                    validity_loss = hinge_loss(validity_logit, valid, existence)
                else:
                    validity_loss = binary_cross_entropy(validity_logit, valid, existence)

                total_loss = validity_loss
                total_loss.backward()
                if flush:
                    self.optimizer.step()
                gs = self.global_step
                self._writer.add_scalar(f'Accuracy/{ds_name.title()}/Validity', fraction(validity_match, existence),  gs)
                self._writer.add_scalar(f'Loss/{ds_name.title()}/Validity', validity_loss, gs)
            else:
                accu = validity_match.sum().cpu().numpy()
                racy = existence.sum().cpu().numpy()
                toke = token_ids.cpu().numpy()
                vali = validity_logit.cpu().numpy()
                offset = offset.cpu().numpy()
                length = length.cpu().numpy()
                self._vis[ds_name][0].process(batch_id, accu, racy, batch_len, offset, length, toke, vali)

        else:
            (batch_size, batch_len, static, dynamic,
             layers_of_exist, layers_of_input,
             layers_of_right, layers_of_valid,
             layers_of_right_gold, layers_of_valid_gold,
             seq_of_pairs, pair_signals, segment,
             seg_length) = self._model(token_ids, offset, length, noise_mode = False, train_clean = mode == M_TRAIN)
            batch_time = time() - batch_time

            valid_logit = layers_of_valid.squeeze(dim = 2)
            right_logit = layers_of_right.squeeze(dim = 2)
            if self._validity_hinge_loss:
                valid = valid_logit > 0
                right = right_logit > 0
            else:
                valid_logit = self._sigmoid(valid_logit)
                right_logit = self._sigmoid(right_logit)
                valid = valid_logit > 0.5
                right = right_logit > 0.5

            layers_of_exist.squeeze_(dim = 2)
            single_unit = segment[None] * (seg_length <= 1)
            single_unit = single_unit.sum(dim = 1)
            right_existence = big_endian_height_mask(right.shape[1], single_unit)
            right_existence &= layers_of_exist

            if mode == M_TRAIN:
                layers_of_valid_gold.squeeze_(dim = 2)
                layers_of_right_gold.squeeze_(dim = 2)
                valid_match = (valid == layers_of_valid_gold)
                right_match = (right == layers_of_right_gold)
                right_match &= right_existence

                seq_of_pairs = seq_of_pairs.squeeze(dim = 1)
                if self._validity_hinge_loss:
                    valid_loss = hinge_loss(valid_logit, layers_of_valid_gold,            None)
                    right_loss = hinge_loss(right_logit, layers_of_right_gold, right_existence)
                    bpe_loss = hinge_loss(seq_of_pairs, pair_signals, None)
                else:
                    valid_loss = binary_cross_entropy(valid_logit, layers_of_valid_gold,            None)
                    right_loss = binary_cross_entropy(right_logit, layers_of_right_gold, right_existence)
                    seq_of_pairs = self._sigmoid(seq_of_pairs)
                    bpe_loss = binary_cross_entropy(seq_of_pairs, pair_signals, None)

                alpha, beta, gamma = self._loss_weights
                total_loss = alpha * valid_loss + beta * right_loss + gamma * bpe_loss
                total_loss.backward()
                if flush:
                    self.optimizer.step()
                gs = self.global_step
                self._writer.add_scalar(f'Accuracy/{ds_name.title()}/Validity',    fraction(valid_match,            None), gs)
                self._writer.add_scalar(f'Accuracy/{ds_name.title()}/Orientation', fraction(right_match, right_existence), gs)
                self._writer.add_scalar(f'Loss/{ds_name.title()}/Validity',                                    valid_loss, gs)
                self._writer.add_scalar(f'Loss/{ds_name.title()}/Orientation',                                 right_loss, gs)
            else:
                valid_score = valid_logit
                right_score = right_logit
                if self._validity_hinge_loss:
                    hinge_score(valid_score, inplace = True)
                    hinge_score(right_score, inplace = True)

                valid_score = torch.where(layers_of_exist, valid_score, torch.ones_like(valid_score))
                split_score = torch.where(right_existence, right_score, torch.ones_like(right_score)) # [0, 1]
                valid_score -= 0.5
                split_score -= 0.5
                valid_score.abs_()
                split_score.abs_() # [0, 0.5]
                valid_score += 0.5
                split_score += 0.5 # [0.5, 1]
                valid_logsum = valid_score.log().sum().cpu().numpy()
                split_logsum = split_score.log().sum().cpu().numpy()
                logsum = valid_logsum, split_logsum
                token = token_ids.cpu().numpy()
                offset = offset.cpu().numpy()
                length = length.cpu().numpy()
                right = right.cpu().numpy()
                valid_score = torch.where(layers_of_exist, valid_score, torch.zeros_like(valid_score)).cpu().numpy()
                segment = segment.cpu().numpy()
                seg_length = seg_length.cpu().numpy()
                vis = self._vis[ds_name][0]
                if vis.save_tensors:
                    right_score = right_score.cpu().numpy()
                    mpc_token = self._model_pca(static).cpu().numpy()
                    mpc_label = self._model_pca(layers_of_input).cpu().numpy()
                    visual = token, offset, length, valid_score, right_score, right, mpc_token, mpc_label, segment, seg_length
                else:
                    visual = token, offset, length, valid_score, None, right, None, None, segment, seg_length
                vis.process(batch_id, batch_len, logsum + visual)

        if mode == M_TRAIN:
            self._writer.add_scalar(f'Loss/{ds_name.title()}/Total',       total_loss, gs)
            self._writer.add_scalar(f'Batch/{ds_name.title()}/Length',      batch_len, gs)
            self._writer.add_scalar(f'Batch/{ds_name.title()}/SamplePerSec', batch_len / batch_time,  gs)

        return batch_size, batch_len

    def _before_validation(self, ds_name, epoch, use_test_set = False, final_test = True):
        self._model.eval()
        if use_test_set:
            folder = f'vis_{ds_name}_test'
            if not final_test:
                folder += '_with_devel'
        else:
            folder = f'vis_{ds_name}_devel'

        if ds_name == D_NOISE:
            vis = ChunkVis(epoch,
                           self.recorder.create_join(folder),
                           self.i2vs,
                           self.recorder.log)
        else:
            # save_tensors = int(float(epoch))
            # save_tensors = save_tensors > 5 and is_bin_times(save_tensors - 5)
            if final_test:
                save_tensors = True
            else:
                save_tensors = is_bin_times(int(float(epoch)) - 1)
            vis = UTreeVis(epoch,
                           self.recorder.create_join(folder),
                           self.i2vs,
                           self.recorder.log,
                           save_tensors)
            self._model_pca = self._model.get_static_pca()
                        
        vis = VisRunner(vis, async_ = True) # TODO: release
        self._vis[ds_name] = vis, use_test_set, final_test
        vis.before()
        
    def _after_validation(self, ds_name, count, seconds):
        vis, use_test_set, final_test = self._vis.pop(ds_name)
        speed = float(f'{count / seconds:.1f}')
        if ds_name == D_NOISE:
            accuracy, ratio = vis.after()
            scores = {'Acc': accuracy, 'speed': speed}
            desc = f'{ds_name}: {accuracy}%'
            if ratio is not None:
                desc += f' | {ratio * 100:.2f}%'
            if not final_test:
                mode = 'TestSet' if use_test_set else 'DevelSet'
                self._writer.add_scalar(f'{mode}/{ds_name.title()}/Acc', accuracy, self.global_step)
        else:
            valid, right = vis.after()
            scores = {'speed': speed, 'log-Val': valid, 'log-Ori': right}
            gs = self.global_step
            if not final_test:
                mode = 'TestSet' if use_test_set else 'DevelSet'
                self._writer.add_scalar(f'{mode}/{ds_name.title()}/log_Val', valid, gs)
                self._writer.add_scalar(f'{mode}/{ds_name.title()}/log_Ori', right, gs)
            desc = f'{ds_name}: log-Val{valid:.2f}, log-Ori{right:.2}'

        self._model.train()
        return scores, desc, desc + f' @{speed}sps.'

    @staticmethod
    def combine_scores_and_decide_key(epoch, ds_scores):
        if D_CLEAN in ds_scores:
            key = log(ds_scores[D_NOISE]['Acc'] / 100)
            key += ds_scores[D_CLEAN]['log-Val']
            key += ds_scores[D_CLEAN]['log-Ori']
        else:
            key = ds_scores[D_NOISE]['Acc']
        ds_scores['key'] = key
        return ds_scores

from utils.vis import BaseVis, VisRunner
from visualization import set_vocab
from collections import Counter
from utils.file_io import join, isfile, listdir, rename
class ChunkVis(BaseVis):
    def __init__(self, epoch, work_dir, i2vs, logger):
        super().__init__(epoch)
        self._work_dir = work_dir
        self._i2vs     = i2vs
        self._logger   = logger
        self._accu = 0
        self._racy = 0
        self._subs = Counter()
        self._tokens = {}
        self._cnts = 0

    def _before(self):
        if self._work_dir:
            set_vocab(self._work_dir, self._i2vs._nested)

    def _process(self, batch_id, accu, racy, size, offsets, lengths, token_ids, validity_logits):
        self._accu += accu
        self._racy += racy

        if self._work_dir is None:
            return
        for offset, length, tokens, logits in zip(offsets, lengths, token_ids, validity_logits):
            tokens = ''.join(self._i2vs.token[t] for t in tokens[offset:offset+length])
            chunks = []
            if length > 2:
                end_threshold = 0.15
                start  = 0
                logits = logits[size:]
                for i in range(0, length - 1):
                    split = False
                    if i == 0: # start
                        mid, right = logits[offset:offset + 2]
                        split = mid + end_threshold < right
                    elif i == length - 2: # end
                        left, mid = logits[offset + i - 1:offset + i + 1]
                        split = left > mid + end_threshold
                    else: # mid
                        left, mid, right = logits[offset + i - 1:offset + i + 2]
                        split = left > mid < right
                    if split:
                        chunk = tokens[start:i + 1]
                        self._subs[chunk] += 1
                        chunks.append(chunk)
                        start = i + 1
                chunks.append(tokens[start:])
                self._tokens[tokens] = '|'.join(chunks)
                self._subs[tokens[start:]] += 1
            else:
                self._subs[tokens] += 1
                self._tokens[tokens] = tokens
            self._cnts += 1

    def _after(self):
        accuracy = float(f'{self._accu / self._racy * 100:.2f}')
        if self._work_dir is None:
            return accuracy, None

        with open(join(self._work_dir, f'{self.epoch}.freq'), 'w') as fw:
            for tok, cnt in sorted(self._subs.items(), key = lambda tc:tc[1], reverse = True):
                fw.write(f'{tok}\t{cnt}\n')

        cfiles = {}
        for fname in listdir(self._work_dir):
            if fname.startswith('chunk.') and fname.endswith('.tsv'):
                cid, cnt = (int(x) for x in fname[6:-4].split('_'))
                cfiles[cid] = cnt

        if len(cfiles) == 0:
            cid, cnt = 0, 0
        else:
            cid = max(cfiles)
            if any(cnt < 10 for cnt in cfiles.values()):
                cnt = cfiles[cid]
            else:
                cid += 1
                cnt = 0
                
        cfile = join(self._work_dir, f'chunk.{cid}_{cnt}.tsv')
        if isfile(cfile):
            with open(cfile, 'r') as fw:
                tokens = {}
                for line in fw:
                    tab = line.index('\t')
                    tok = line[:tab]
                    val = line[tab + 1:-1] + '\t' + self._tokens[tok]
                    self._tokens[tok] = val

        with open(cfile, 'w') as fw:
            for tok, chunks in self._tokens.items():
                fw.write(f'{tok}\t{chunks}\n')
        rename(cfile, join(self._work_dir, f'chunk.{cid}_{cnt + 1}.tsv'))

        return accuracy, len(self._subs) / self._cnts

from visualization import set_void_head, set_data
class UTreeVis(BaseVis):
    def __init__(self, epoch, work_dir, i2vs, logger, save_tensors):
        super().__init__(epoch)
        self._work_dir = work_dir
        self._i2vs     = i2vs
        self._logger   = logger
        self._valid_log_score = 0
        self._right_log_score = 0
        self._data_tree = None
        self._head_init = None
        self.register_property('save_tensors', save_tensors)

    def _before(self):
        if self._work_dir:
            self._head_init = set_vocab(self._work_dir, self._i2vs._nested)
            self._data_tree = open(join(self._work_dir, f'data.{self.epoch}.tree'), 'w')
    
    def _process(self, batch_id, size, batch):

        (valid_logsum, right_logsum,
         token, offset, length, valid_score, right_score, right, mpc_token, mpc_label,
         segment, seg_length) = batch
        self._valid_log_score += valid_logsum
        self._right_log_score += right_logsum
        # import pdb; pdb.set_trace()

        if self._head_init:
            set_void_head(self._work_dir, batch_id, size, offset, length, token)

        warnings = set_data(self._work_dir if self.save_tensors else None, batch_id, size, self.epoch,
                            offset, length, token, None, None, right, mpc_token, mpc_label, None,
                            valid_score, right_score, (segment, seg_length), self._i2vs, self._data_tree)

    def _after(self):
        valid_log_score = float(self._valid_log_score)
        right_log_score = float(self._right_log_score)
        if self._data_tree: self._data_tree.close()
        return valid_log_score, right_log_score

    def __del__(self):
        if self._data_tree: self._data_tree.close()