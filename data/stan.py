import tensorflow as tf
from os.path import join
import numpy as np
from data.io import BOS, EOS, load_vocabs, get_fasttext, TreeBatch
from data.delta import ROB, LOE, xtype_to_logits, LogitX
from data.backend import tf_vocabs_joint, tf_load_dataset, filter_and_shuffle
from data.backend import bucket_and_batch, WordBaseReader, word_batch_seq

class StanReader(WordBaseReader):
    def __init__(self,
                 vocab_dir,
                 categorical_syn,
                 main_tv = tuple()):
        vocabs = load_vocabs(vocab_dir, 'word polar'.split())
        with tf.name_scope('stan_vocab'):
            vocabs = tf_vocabs_joint(vocabs, main_tv, categorical_syn)
        pre_trained = get_fasttext(join(vocab_dir, 'word.vec'))
        if vocabs[-1]: # extra_id
            idx = np.asarray(vocabs[-1])
            pre_trained = pre_trained[idx]
        super(StanReader, self).__init__(vocab_dir, vocabs, pre_trained)

    def tf_input_batch(self,
                       mode,
                       batch_size,
                       num_buckets,
                       shuffle_buf_size = None,
                       shuffle_seed     = None):
        if mode not in ('train', 'eval', 'test'):
            raise ValueError('dataset should be in train, eval or test')

        vocabs = self.vocabs
        itype, _ = self.dtypes
        if self.vocabs.tag: # categorical
            s_paddings = (BOS, EOS)
        else:
            s_paddings = ('2', '2')
        x_paddings = xtype_to_logits(ROB, to_str = False), xtype_to_logits(LOE, to_str = False)
        
        with tf.name_scope('data_' + mode):
            _, weos, wd, nw = tf_load_dataset(self.dir_join('%s.word'  % mode), vocabs[0], (BOS, EOS), True,  itype)
            _, seos, sd, ns = tf_load_dataset(self.dir_join('%s.polar' % mode), vocabs[2], s_paddings, False, itype)
            _, xeos, xd, nx = tf_load_dataset(self.dir_join('%s.xtype' % mode),      None, x_paddings, False, itype)

            if nw == ns == nx:
                num_sample = nw
                if shuffle_buf_size == 0:
                    shuffle_buf_size = num_sample
            else:
                raise ValueError('Corpus is not aligned')

            data = tf.data.Dataset.zip((wd, sd, xd))
            data = filter_and_shuffle(data, 0, None, shuffle_buf_size, batch_size, shuffle_seed)
            data = bucket_and_batch(data, (weos, seos, xeos), batch_size, num_buckets)
            batched_iter = data.make_initializable_iterator()

            seq_len, word_ids, polar_ids, xtype_ids = batched_iter.get_next()
            batch_size_, seq_size_ = word_batch_seq(word_ids)
            logits = self.split_xtype(xtype_ids)

        return {'stan_batch': TreeBatch(
                    initializer = batched_iter.initializer,
                    reinit_rand = lambda: None,
                    word        = word_ids,
                    tag         = None,
                    label       = polar_ids,
                    ftag        = None,
                    finc        = None,
                    seq_len     = seq_len,
                    num_sample  = num_sample,
                    seq_size    = seq_size_,
                    batch_size  = batch_size_,
                    full_batch_size = batch_size,
                    **{'ori_' + k:v for k,v in zip(LogitX._fields, logits)}),
                'stan_specs': self.specs}

if __name__ == '__main__':
    from data.delta import cky_to_triangle
    from sys import stderr

    def pad(s, n):
        h = n >> 1
        return [BOS] * h + s + [EOS] * h

    def zip_iter(_syn, _xty, exclude_paddings):
        with open(_syn, 'r') as fs,\
             open(_xty, 'r') as fx:
            for s_line, x_line in tqdm(zip(fs, fx)):
                print(len(s_line.split()), len(x_line.split()))
                # s = cky_to_triangle(s_line.split(), exclude_paddings)
                # x = cky_to_triangle(x_line.split(), exclude_paddings)
                # yield s, x
                yield s_line.split(), x_line.split()

    def peek_tree(_syn, _xty, i):
        cnt = 0
        for s, x in zip_iter(_syn, _xty, True):
            if cnt == i:
                print('\t'.join('%s:%s'%k for k in zip(x.pop(0), p)))
                for xl, sl in zip(x,s):
                    print('\t'.join('%s:%s'%k for k in zip(xl, sl)))
                return
            cnt += 1

    try:
        workdir, prefix, n = sys.argv[1:]
        _syn = join(workdir, '%s.pol_syn' % prefix)
        _xty = join(workdir, '%s.pol_xtype' % prefix)
        peek_tree(_syn, _xty, int(n))
    except ValueError as e:
        print(e, file = sys.stderr)
        print('Useage: this_file.py [peek_n_gram|peek_tree] path_to_data prefix(train/eval/eval/test) odd_integer|id > _print.txt', file = sys.stderr)
        exit()
