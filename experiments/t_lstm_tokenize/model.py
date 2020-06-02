from models.backend import InputLeaves, Contextual, input_config, contextual_config
from models.backend import Stem, stem_config
from models.backend import torch, nn, Tensor, activation_type
from models.utils import condense_helper, condense_left, bos_mask, eos_mask, shuffle_some_from
from utils.types import word_dim, orient_dim, frac_2
from math import log

input_config = input_config.copy()
input_config.pop('pre_trained')
input_config.pop('trainable')
discrim_config = dict(hidden_dim = orient_dim, activation = activation_type, drop_out = frac_2)
tokenizer_config = dict(model_dim        = word_dim,
                        input_layer      = input_config,
                        contextual_layer = contextual_config,
                        discrim_layer    = discrim_config,
                        orient_layer     = stem_config)

class VectorDiscriminator(nn.Module):
    def __init__(self,
                 model_dim,
                 hidden_dim,
                 activation,
                 drop_out):
        super().__init__()
        
        self._to_hidden = nn.Linear(model_dim, hidden_dim)
        self._to_logit = nn.Linear(hidden_dim, 1)
        self._activation = activation()
        self._drop_out = nn.Dropout(drop_out)

    def forward(self, vectors):
        hidden = self._to_hidden(vectors)
        hidden = self._drop_out(hidden)
        hidden = self._activation(hidden)
        return self._to_logit(hidden)

class RnnTokenizer(nn.Module):
    def __init__(self,
                 num_tokens,
                 paddings,
                 model_dim,
                 input_layer,
                 contextual_layer,
                 discrim_layer,
                 orient_layer):
        super().__init__()

        input_layer['pre_trained'] = False
        input_layer['trainable']   = True
        self._input_layer = InputLeaves(model_dim, num_tokens, None, **input_layer)
        self._contextual_layer = Contextual(model_dim, 'hidden_never_used', **contextual_layer)
        self._discriminator = VectorDiscriminator(model_dim, **discrim_layer)
        self._discr_sigmoid = nn.Sigmoid()
        self._orient_layer = Stem(model_dim, **orient_layer)
        self._paddings = paddings

    def forward(self, char_idx, offset, length, noise_mode, train_clean = False, **kw_args):
        batch_size, batch_len = char_idx.shape
        static, bottom_existence = self._input_layer(char_idx)
        dynamic, _ = self._contextual_layer(static)

        if dynamic is None:
            bottom_input = static
        else:
            bottom_input = dynamic

        bottom_existence.unsqueeze_(dim = 2)
        base_return = batch_size, batch_len, static, dynamic

        if noise_mode:
            second_existence, second_input = self._orient_layer.blind_combine(bottom_input, bottom_existence)
            existence  = torch.cat([bottom_existence, second_existence], dim = 1)
            dual_input = torch.cat([bottom_input,         second_input], dim = 1)
            validity   = self._discriminator(dual_input)
            return base_return + (existence, validity)

        dtype = static.dtype
        min_val = torch.finfo(dtype).min
        frac_pad = torch.ones(batch_size, 1, 1, dtype = dtype, device = static.device) * min_val
        layers_of_exist = []
        layers_of_input = []
        layers_of_right = []
        layers_of_valid = [self._discriminator(bottom_input)]
        layers_of_right_gold = []
        layers_of_valid_gold = [bottom_existence]
        seq_of_pairs = []
        pair_signals = []
        segments    = []
        seg_lengths = []
        h0c0 = self._orient_layer.get_h0c0(batch_size)
        valid_gold = lw_relay = rw_relay = None
        batch_range = torch.arange(batch_size, device = static.device)

        while True:
            pred_orient = self._orient_layer.predict_orient(bottom_input, h0c0) # should learn from validity with probability
            layers_of_exist.append(bottom_existence)
            layers_of_input.append(bottom_input)
            segments.append(bottom_input.shape[1])
            seg_length = bottom_existence.sum(dim = 1)
            seg_lengths.append(seg_length)
            if torch.all(seg_length <= 1):
                layers_of_right.append(pred_orient)
                if train_clean:
                    layers_of_right_gold.append(pred_orient > 0)
                break
            
            if train_clean:
                _, test_input = self._orient_layer.blind_combine(bottom_input, bottom_existence)
                next_validity = self._discriminator(test_input) # nil_pads are invalid
                next_validity = next_validity * bos_mask(bottom_input.shape[1] - 1, (seg_length - 1).squeeze()).unsqueeze(dim = 2)

                padded_validity = torch.cat([frac_pad, next_validity, frac_pad], dim = 1)
                if True:
                    right = torch.exp(padded_validity[:, 1:])
                    total = torch.exp(padded_validity[:, :-1]) + right # left + right
                    prob_ = right / total
                    right = torch.rand_like(prob_) < prob_ # pseudo-gold right
                else:
                    right = padded_validity[:, :-1] < padded_validity[:, 1:]
                # TODO:
                # dbg |> test stability & confidence | PPL of left and right; them break
                # noise bpe
                shuffled, continuous = shuffle_some_from(batch_size * batch_len, bottom_input, bottom_existence)
                shuffled = self._orient_layer.blind_combine(shuffled)
                shuffled = self._discriminator(shuffled)
                seq_of_pairs.append(shuffled)
                pair_signals.append(continuous)
            else:
                right = pred_orient > 0 # determined non-train
                
            if lw_relay is not None:
                right[lw_relay] = False
                right[rw_relay] = True
                right[batch_range, 0] = True
                eos = eos_mask(right.shape[1], seg_length - 1)
                eos.transpose_(1, 2)
                right[eos] = False
            else:
                right[bos_mask(batch_len, offset + 1)] = True
                right[eos_mask(batch_len, offset + length - 1)] = False

            (next_existence, new_jnt, lw_relay, rw_relay,
             next_input) = self._orient_layer.combine(right, bottom_input, bottom_existence)

            layers_of_right.append(pred_orient)
            if train_clean: # next layer
                # new_jnt = new_jnt | (torch.rand_like(new_jnt, dtype = dtype) < 0.2)
                # import pdb; pdb.set_trace()
                valid_gold = torch.rand_like(next_validity) < self._discr_sigmoid(next_validity) * 0.9
                valid_gold[~next_existence] = False
                # valid_gold[new_jnt] = True
                if valid_gold is not None:
                    # TODO put ratio here, probabilistic loss/penalty
                    ends = next_existence.sum(dim = 1)
                    ends.squeeze_(dim = 1)
                    ends -= 1 # left and upper single
                    valid_gold[batch_range, ends] = True

                layers_of_valid.append(next_validity)
                layers_of_valid_gold.append(valid_gold)
                layers_of_right_gold.append(right)

            next_existence.squeeze_(dim = 2)
            next_helper = condense_helper(next_existence, as_existence = True)
            bottom_input, bottom_existence = condense_left(next_input, next_helper, get_cumu = True)
            # import pdb; pdb.set_trace()
            lw_relay = condense_left(lw_relay, next_helper)
            rw_relay = condense_left(rw_relay, next_helper)

        # print('segment', segments)
        # print('seg_length', ', '.join(str(x.shape[1]) for x in seg_lengths))
        # print('exist     ', ', '.join(str(x.shape[1]) for x in layers_of_exist))
        # print('input     ', ', '.join(str(x.shape[1]) for x in layers_of_input))
        # print('right     ', ', '.join(str(x.shape[1]) for x in layers_of_right))
        # print('right_gold', ', '.join(str(x.shape[1]) for x in layers_of_right_gold))
        # print('valid     ', ', '.join(str(x.shape[1]) for x in layers_of_valid))
        # print('valid_gold', ', '.join(str(x.shape[1]) for x in layers_of_valid_gold))
        
        segments   .reverse()
        seg_lengths.reverse()
        layers_of_exist.reverse()
        layers_of_input.reverse()
        layers_of_right.reverse()
        segment    = torch.tensor(segments, dtype = seg_length.dtype, device = seg_length.device)
        seg_length = torch.cat(seg_lengths, dim = 1)
        layers_of_exist = torch.cat(layers_of_exist, dim = 1)
        layers_of_input = torch.cat(layers_of_input, dim = 1)
        layers_of_right = torch.cat(layers_of_right, dim = 1)
        if train_clean:
            # import pdb; pdb.set_trace()
            layers_of_valid.reverse()
            layers_of_right_gold.reverse()
            layers_of_valid_gold.reverse()
            layers_of_valid = torch.cat(layers_of_valid, dim = 1)
            layers_of_right_gold = torch.cat(layers_of_right_gold, dim = 1)
            layers_of_valid_gold = torch.cat(layers_of_valid_gold, dim = 1)
            seq_of_pairs = torch.cat(seq_of_pairs, dim = 0)
            pair_signals = torch.cat(pair_signals, dim = 0)
        else:
            layers_of_valid = self._discriminator(layers_of_input)

        clean_return = (layers_of_exist, layers_of_input,
                        layers_of_right, layers_of_valid,
                        layers_of_right_gold, layers_of_valid_gold,
                        seq_of_pairs, pair_signals,
                        segment, seg_length)
        return base_return + clean_return

    def get_static_pca(self):
        self._input_layer.flush_pc()
        return self._input_layer.pca