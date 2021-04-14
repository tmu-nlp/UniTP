from experiments.t_lstm_nccp import PennOperator, train_type
from data.stan_types import C_SSTB
from utils.types import M_TRAIN, rate_5, NIL
from time import time
from models.utils import PCA, fraction, hinge_score, torch
from models.loss import binary_cross_entropy, hinge_loss
from data.delta import get_rgt, get_dir, s_index
from utils.math_ops import is_bin_times, f_score
from utils.shell_io import byte_style
from sys import stderr

train_type = train_type.copy()
train_type['loss_weight'] = train_type['loss_weight'].copy()
train_type['loss_weight']['sentiment_orient'] = rate_5
train_type['loss_weight']['sentiment_label']  = rate_5

class StanOperator(PennOperator):
    def __init__(self, model, get_datasets, recorder, penn_i2vs, sstb_i2vs, evalb, train_config):
        super().__init__(model, get_datasets, recorder, penn_i2vs, evalb, train_config)
        self._stan_i2vs = sstb_i2vs

    def _step(self, mode, ds_name, batch, batch_id = None):
        if ds_name == C_SSTB:
            gold_orients = get_rgt(batch['xtype'])
            # if mode == M_TRAIN:
            batch['supervised_orient'] = gold_orients
            batch['is_sentiment'] = True

            batch_time = time()
            (batch_size, batch_len, static, top3_polar_logits,
             layers_of_base, _, existences, orient_logits, _, _, trapezoid_info,
             polar_logits) = self._model(batch['token'], self._tune_pre_trained, **batch)
            batch_time = time() - batch_time

            orient_logits.squeeze_(dim = 2)
            existences   .squeeze_(dim = 2)
            if self._train_config.orient_hinge_loss:
                orients = orient_logits > 0
            else:
                orient_logits = self._sigmoid(orient_logits)
                orients = orient_logits > 0.5

            if mode == M_TRAIN:
                polars = self._model.get_polar_decisions(polar_logits)
                orient_weight = get_dir(batch['xtype'])
                orient_match  = (orients == gold_orients) & orient_weight
                polar_mis     = (polars[:, :, 0]  != batch['polar'])
                polar_weight  = (polar_mis | existences)

                if trapezoid_info is None:
                    height_mask = s_index(batch_len - batch['length'])[:, None, None]
                else:
                    height_mask = batch['mask_length'] # ?? negative effect ???

                polar_loss = self._model.get_polar_loss(polar_logits, top3_polar_logits, batch, height_mask)

                if self._train_config.orient_hinge_loss:
                    orient_loss = hinge_loss(orient_logits, gold_orients, orient_weight)
                else:
                    orient_loss = binary_cross_entropy(orient_logits, gold_orients, orient_weight)

                total_loss = self._train_config.loss_weight.sentiment_label * polar_loss
                total_loss = self._train_config.loss_weight.sentiment_orient * orient_loss + total_loss
                total_loss.backward()

                gs = self.global_step
                self.recorder.tensorboard(gs, 'Accuracy/%s',
                                          Polar = 1 - fraction(polar_mis,  polar_weight),
                                          PolarOrient = fraction(orient_match, orient_weight))

                self.recorder.tensorboard(gs, 'Loss/%s', Polar = polar_loss, PolarOrient = orient_loss, PolarTotal = total_loss)
                batch_kwargs = dict(PolarLength = batch_len, PolarSamplePerSec = batch_len / batch_time)
                if 'segment' in batch:
                    batch_kwargs['PolarHeight'] = len(batch['segment'])
                self.recorder.tensorboard(self.global_step, 'Batch/%s', **batch_kwargs)
            else:
                vis, _, _ = self._vis_mode
                if vis.save_tensors:
                    pca = self._model.get_static_pca()
                    if pca is None:
                        try:
                            pca = PCA(layers_of_base.reshape(-1, layers_of_base.shape[2]))
                            mpc_token = pca(static)
                            mpc_label = pca(layers_of_base)
                            b_mpcs = (mpc_token.type(torch.float16), mpc_label.type(torch.float16))
                        except RuntimeError:
                            b_mpcs = 'RuntimeError catched:\n'
                            b_mpcs += 'Maybe you have used an activation function without any negative output,\n'
                            b_mpcs += '  e.g. ReLU, Softplus, and Sigmoid!\n'
                            b_mpcs += 'Visualization is disabled, while (bad) results keep generating.'
                            print(byte_style(b_mpcs, '1'), file = stderr)
                            b_mpcs = (None, None)
                    else:
                        mpc_token = pca(static)
                        mpc_label = pca(layers_of_base)
                        b_mpcs = (mpc_token.type(torch.float16), mpc_label.type(torch.float16))

                    polar_scores, polars = self._model.get_polar_decisions_with_values(polar_logits)
                    if self._train_config.orient_hinge_loss: # otherwise with sigmoid
                        hinge_score(orient_logits, inplace = True)
                    b_scores = (polar_scores.type(torch.float16), orient_logits.type(torch.float16))
                else:
                    polars = self._model.get_polar_decisions(polar_logits)
                    b_mpcs = b_scores = (None, None)

                b_size = (batch_len,)
                b_head = tuple(batch[x] for x in 'offset length token'.split())
                b_pola = batch['polar'].type(torch.int8)
                polars = polars.type(torch.int8)
                if NIL not in self._stan_i2vs.polar: # nil_is_neutral
                    # import pdb; pdb.set_trace()
                    polars[~existences] = -1
                    b_pola[~existences] = -1
                b_head = b_head + (b_pola, gold_orients)
                b_logits = polars, orients
                b_data = b_logits + b_mpcs + b_scores
                tensors = b_size + b_head + b_data
                tensors = tuple(x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in tensors)
                if trapezoid_info is not None:
                    trapezoid_info = batch['segment'], batch['seg_length'] #, d_seg, d_seg_len.cpu().numpy()
                vis.process(batch_id, tensors, trapezoid_info)
            return batch_size, batch_len
        return super()._step(mode, ds_name, batch, batch_id)

    def _before_validation(self, ds_name, epoch, use_test_set = False, final_test = False):
        if ds_name != C_SSTB:
            return super()._before_validation(ds_name, epoch, use_test_set, final_test)

        devel_bins, test_bins = self._mode_length_bins
        if use_test_set:
            if final_test:
                folder = ds_name + '_test'
            else:
                folder = ds_name + '_test_with_devel'
            save_tensors = True
            length_bins = test_bins
            scores_of_bins = True
        else:
            folder = ds_name + '_devel'
            length_bins = devel_bins
            save_tensors = is_bin_times(int(float(epoch)) - 1)
            scores_of_bins = False
            
        self._model.update_static_pca()
        vis = StanVis(epoch,
                      self.recorder.create_join(folder),
                      self._stan_i2vs,
                      self.recorder.log,
                      save_tensors,
                      length_bins,
                      scores_of_bins,
                      final_test)
        vis = VisRunner(vis, async_ = True) # wrapper
        vis.before()
        length_bins = vis.length_bins
        if length_bins is not None:
            if use_test_set:
                self._mode_length_bins = devel_bins, length_bins # change test
            else:
                self._mode_length_bins = length_bins, test_bins # change devel
        self._vis_mode = vis, use_test_set, final_test

    def _after_validation(self, ds_name, count, seconds):
        if ds_name != C_SSTB:
            return super()._after_validation(ds_name, count, seconds)
        vis, use_test_set, final_test = self._vis_mode
        scores, desc, logg = vis.after()
        speed = float(f'{count / seconds:.1f}')
        if vis.is_async:
            rate = vis.proc_time / seconds
        else:
            rate = vis.proc_time / (seconds - vis.proc_time)
        logg += f' @{speed}sps. (sym:nn {rate:.2f})'
        if not final_test:
            self.recorder.tensorboard(self.global_step, 'TestSet/%s' if use_test_set else 'DevelSet/%s', **scores)
        scores['speed'] = speed
        self._vis_mode = None
        return scores, desc, logg

    @staticmethod
    def combine_scores_and_decide_key(epoch, ds_scores):
        scores = ds_scores[C_SSTB]
        qui = f_score(scores['5'], scores['Q'], 2)
        ter = f_score(scores['3'], scores['T'], 2) #∴⋮:
        bi  = f_score(scores['2'], scores['B'], 2)
        ds_scores['key'] = f_score(f_score(bi, ter, 2), qui, 2)
        return ds_scores

from utils.vis import BaseVis, VisRunner
from utils.file_io import join, isfile, listdir, remove, isdir
from utils.pickle_io import pickle_dump
from utils.param_ops import HParams
from utils.shell_io import parseval, rpt_summary, byte_style
from visualization import ContinuousTensorVis
from data.stan_types import calc_stan_accuracy
class StanVis(BaseVis):
    def __init__(self, epoch, work_dir, i2vs, logger,
                 save_tensors   = True,
                 length_bins    = None,
                 scores_of_bins = False,
                 flush_heads    = False):
        super().__init__(epoch)
        fname = join(work_dir, 'vocabs.pkl') # TODO integrate into ctvis
        if flush_heads and isfile(fname):
            remove(fname)
        self._ctvis = ContinuousTensorVis(work_dir, i2vs)
        self._logger = logger
        htree = join(work_dir, 'head.tree')
        dtree = join(work_dir, f'data.{epoch}.tree')
        self._fnames = htree, dtree
        self._head_tree = None
        self._data_tree = None
        self._scores_of_bins = scores_of_bins
        self.register_property('save_tensors', save_tensors)
        self.register_property('length_bins',  length_bins)
        self._final_dn = None

    def __del__(self):
        if self._head_tree: self._head_tree.close()
        if self._data_tree: self._data_tree.close()

    def _before(self):
        htree, dtree = self._fnames
        if self._ctvis.is_anew: # TODO
            self._head_tree = open(htree, 'w')
            self.register_property('length_bins', set())
        self._data_tree = open(dtree, 'w')

    def _process(self, batch_id, batch, trapezoid_info):
        (size, h_offset, h_length, h_token, h_polar, h_right,
         d_polar, d_right, mpc_token, mpc_label,
         label_score, split_score) = batch

        if self._head_tree:
            bins = self._ctvis.set_head(self._head_tree, h_offset, h_length, h_token, None, h_polar, h_right, trapezoid_info, batch_id, size, 10)
            self.length_bins |= bins

        if self.length_bins is not None and self._scores_of_bins:
            bin_width = 10
        else:
            bin_width = None

        if self.save_tensors and (mpc_token is not None or mpc_label is not None):
            if self.length_bins is not None and self._scores_of_bins:
                bin_width = 10 # TODO check this!!!
            else:
                bin_width = None
            extended = size, bin_width, None
        else:
            extended = None

        n__d = self._ctvis.set_data(self._data_tree, self._logger, batch_id, self.epoch,
                             h_offset, h_length, h_token, None, d_polar, h_right, # TODO d_right, should be equal
                             mpc_token, mpc_label,
                             None, label_score, split_score,
                             trapezoid_info,
                             extended)

        if n__d is not None:
            n, d = n__d
            if self._final_dn is None:
                self._final_dn = n, d
            else:
                fn, fd = self._final_dn
                fn += n
                fd += d

    def _after(self):
        if self._head_tree:
            self._head_tree.close()
        self._data_tree.close()
        
        if self._final_dn is None:
            fn, fd = calc_stan_accuracy(*self._fnames, self.epoch, self._logger)[-1]
        else:
            fn, fd = self._final_dn
        scores = {d: float(f'{f * 100:.2f}') for f, d in zip(fn / fd, ('532QTB'))}
        desc_0 = f'☺︎({scores["B"]:.0f}\'{scores["T"]:.0f}\'{scores["Q"]:.0f}|'
        desc_1 = f'{scores["2"]:.0f}\'{scores["3"]:.0f}\'{scores["5"]:.0f}'
        logg = desc_0 + desc_1 + ')'
        desc = desc_0 + byte_style(desc_1, underlined = True) + ')'
        return scores, desc, logg
