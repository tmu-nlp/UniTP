from experiments.t_lstm_parse import PennOperator, train_type
from data.stan_types import C_SSTB
from utils.types import M_TRAIN, rate_5, NIL
from time import time
from models.utils import PCA, fraction, hinge_score, torch
from models.loss import binary_cross_entropy, hinge_loss, cross_entropy
from data.delta import get_rgt, get_dir, s_index
from utils.math_ops import is_bin_times, f_score

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
                
            batch_time = time()
            (batch_size, batch_len, static, dynamic, top3_polar_logits,
             layers_of_base, _, existences, orient_logits, _, _, trapezoid_info,
             polar_logits) = self._model(batch['token'],
                                         tune_xlnet   = self._tune_xlnet,
                                         is_sentiment = True, **batch)
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

                polar_loss = self._model.get_polar_loss(polar_logits, batch, height_mask)
                if top3_polar_logits is not None:
                    polar_loss += cross_entropy(top3_polar_logits, batch['top3_polar'], None)

                if self._train_config.orient_hinge_loss:
                    orient_loss = hinge_loss(orient_logits, gold_orients, orient_weight)
                else:
                    orient_loss = binary_cross_entropy(orient_logits, gold_orients, orient_weight)

                total_loss = self._train_config.loss_weight.sentiment_label * polar_loss
                total_loss = self._train_config.loss_weight.sentiment_orient * orient_loss + total_loss
                total_loss.backward()
                gs = self.global_step
                self._writer.add_scalar('Accuracy/Polar',       1 - fraction(polar_mis,  polar_weight), gs)
                self._writer.add_scalar('Accuracy/PolarOrient',  fraction(orient_match, orient_weight), gs)
                self._writer.add_scalar('Loss/Polar',        polar_loss,  gs)
                self._writer.add_scalar('Loss/PolarOrient',  orient_loss, gs)
                self._writer.add_scalar('Loss/PolarTotal',   total_loss,  gs)
                self._writer.add_scalar('Batch/PolarSamplePerSec', batch_len / batch_time,  gs)
                self._writer.add_scalar('Batch/PolarLength', batch_len,   gs)
                if 'segment' in batch:
                    self._writer.add_scalar('Batch/PolarHeight', len(batch['segment']), gs)
            else:
                vis, _, _ = self._vis_mode
                mpc_token = mpc_label = None
                if vis.save_tensors:
                    if hasattr(self._model._input_layer, 'pca'):
                        if dynamic is not None: # even dynamic might be None, being dynamic is necessary to train a good model
                            mpc_token = self._model._input_layer.pca(static)
                        mpc_label = self._model._input_layer.pca(layers_of_base)
                    else:
                        mpc_label = PCA(layers_of_base[:, -batch_len:].reshape(-1, layers_of_base.shape[2]))(layers_of_base)

                    polar_scores, polars = self._model.get_polar_decisions_with_values(polar_logits)
                    if self._train_config.orient_hinge_loss: # otherwise with sigmoid
                        hinge_score(orient_logits, inplace = True)
                    b_mpcs = (None if mpc_token is None else mpc_token.type(torch.float16), mpc_label.type(torch.float16))
                    b_scores = (polar_scores.type(torch.float16), orient_logits.type(torch.float16))
                else:
                    polars = self._model.get_polar_decisions(polar_logits)
                    b_mpcs = (mpc_token, mpc_label)
                    b_scores = (None, None)

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
                folder = 'sstb_test'
            else:
                folder = 'sstb_test_with_devel'
            save_tensors = True
            length_bins = test_bins
            scores_of_bins = True
        else:
            folder = 'sstb_devel'
            length_bins = devel_bins
            save_tensors = is_bin_times(int(float(epoch)) - 1)
            scores_of_bins = False
            
        vis = StanVis(epoch,
                      self.recorder.create_join(folder),
                      self._stan_i2vs,
                      self.recorder.log,
                      save_tensors,
                      length_bins,
                      scores_of_bins)
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
        scores['speed'] = speed
        if not final_test:
            mode = 'TestSet' if use_test_set else 'DevelSet'
            self._writer.add_scalar(f'{mode}/F1', scores.get('F1', 0), self.global_step)
            self._writer.add_scalar(f'{mode}/SamplePerSec', speed,     self.global_step)
        self._vis_mode = None
        return scores, desc, logg

    @staticmethod
    def combine_scores_and_decide_key(epoch, ds_scores):
        scores = ds_scores[C_SSTB]
        qui = f_score(scores['5'], scores['*'], 2)
        ter = f_score(scores['3'], scores['∴'], 2)
        bi  = f_score(scores['2'], scores[':'], 2)
        scores['key'] = f_score(f_score(bi, ter, 2), qui, 2)
        return scores

from utils.vis import BaseVis, VisRunner
from utils.file_io import join, isfile, listdir, remove, isdir
from utils.pickle_io import pickle_dump
from utils.param_ops import HParams
from utils.shell_io import parseval, rpt_summary
from visualization import set_vocab, set_head, set_data, calc_stan_accuracy
class StanVis(BaseVis):
    def __init__(self, epoch, work_dir, i2vs, logger,
                 save_tensors   = True,
                 length_bins    = None,
                 scores_of_bins = False):
        super().__init__(epoch)
        self._work_dir = work_dir
        self._i2vs = i2vs
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
        if set_vocab(self._work_dir, self._i2vs._nested):
            self._head_tree = open(htree, 'w')
            self.register_property('length_bins', set())
        self._data_tree = open(dtree, 'w')

    def _process(self, batch_id, batch, trapezoid_info):
        (size, h_offset, h_length, h_token, h_polar, h_right,
         d_polar, d_right, mpc_token, mpc_label,
         label_score, split_score) = batch

        if self._head_tree:
            bins = set_head(self._work_dir, batch_id,
                            size, h_offset, h_length, h_token, None, h_polar, h_right,
                            trapezoid_info,
                            self._i2vs, self._head_tree)
            self.length_bins |= bins

        if self.length_bins is not None and self._scores_of_bins:
            bin_width = 10
        else:
            bin_width = None

        fpath = self._work_dir if self.save_tensors else None
        n__d = set_data(fpath, batch_id, size, self.epoch,
                        h_offset, h_length, h_token, None, d_polar, h_right, # d_right,
                        mpc_token, mpc_label,
                        None, label_score, split_score,
                        trapezoid_info,
                        self._i2vs, self._data_tree, self._logger, None, bin_width)
        if n__d is not None:
            n, d = n__d
            if self._final_dn is None:
                self._final_dn = n, d
            else:
                fn, fd = self._final_dn
                fn += n
                fd += d

    def _after(self):
        # call evalb to data.emm.rpt return the results, and time counted
        # provide key value in results
        if self._head_tree:
            self._head_tree.close()
        self._data_tree.close()
        
        # fname = None
        # if num_errors:
        #     self._logger(f'  {num_errors} errors from evalb')
        #     if num_errors < 10:
        #         for e, error in enumerate(errors):
        #             self._logger(f'    {e}. ' + error)
        #         fname = f'data.{self.epoch}.rpt'


        # if self.length_bins is not None and self._scores_of_bins:
        #     fname = f'data.{self.epoch}.rpt'
        #     with open(join(self._work_dir, f'{self.epoch}.scores'), 'w') as fw:
        #         fw.write('wbin,num,lp,lr,f1,ta\n')
        #         for wbin in self.length_bins:
        #             fhead = join(self._work_dir, f'head.bin_{wbin}.tree')
        #             fdata = join(self._work_dir, f'data.bin_{wbin}.tree')
        #             proc = parseval(self._evalb, fhead, fdata)
        #             smy = rpt_summary(proc.stdout.decode(), False, True)
        #             fw.write(f"{wbin},{smy['N']},{smy['LP']},{smy['LR']},{smy['F1']},{smy['TA']}\n")
        #             remove(fhead)
        #             remove(fdata)
        if self._final_dn is None:
            fn, fd = calc_stan_accuracy(*self._fnames, self.epoch, self._logger)[-1]
        else:
            fn, fd = self._final_dn
        scores = {d: float(f'{f * 100:.2f}') for f, d in zip(fn / fd, ('532*∴:'))}
        desc = f'☺︎({scores["*"]:.0f}\'{scores["∴"]:.0f}\'{scores[":"]:.0f}|'
        desc += f'{scores["5"]:.0f}\'{scores["3"]:.0f}\'{scores["2"]:.0f})'
        return scores, desc, desc
