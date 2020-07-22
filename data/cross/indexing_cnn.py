from models.backend import torch, nn
from models.utils import condense_splitter, condense_left

def interweave(right_layer, joint_layer, existence):
    batch_size, batch_len = joint_layer.shape
    r0 = right_layer[:, 0, None]
    e0 = existence  [:, 0, None]
    jc = joint_layer[:, :, None]
    ec = existence  [:, 1:, None]
    r1 = right_layer[:, 1:, None]
    jer = torch.cat([jc, ec, r1], dim = 2)
    jer = jer.reshape(batch_size, batch_len * 3)
    jerj = torch.cat([e0, r0, jer], dim = 1) # erj[er]jer
    return jerj.unsqueeze(dim = 1)

class Index(nn.Module):
    def __init__(self,
                 half_window  = 2,
                 fine_adjust  = -1,
                 num_channels = 8,
                 dtype        = torch.int64):
        super().__init__()
        self._dtype = dtype
        self._repr = f'(±{half_window}{fine_adjust:+d}x{num_channels}ch)' # 2x6 should be minimun, 2x8 is safer
        win_len = 6 * half_window + 2 + 2 * fine_adjust
        self._cnn = nn.Conv1d(1, num_channels, win_len, stride = 3, padding = 3 * half_window + fine_adjust) #, bias = False)
        self._act = nn.Tanh()
        self._dpl = nn.Dropout(0.01)

    def forward(self, right_layer, joint_layer, existence):
        right_layer = right_layer * 2.0
        joint_layer = joint_layer * 2.0
        existence   = existence   * 2.0
        right_layer -= 1.0
        joint_layer -= 1.0
        existence   -= 1.0
        layer = interweave(right_layer, joint_layer, existence)
        layer = self._dpl(layer) # pre is much much better than post
        channels = self._cnn(layer)
        return self._act(channels).sum(dim = 1)

    def diff_integer_indice(self, right_layer, joint_layer, existence):
        is_training = self.training
        self.eval() # turn dropout off
        with torch.no_grad():
            indices = self.forward(right_layer, joint_layer, existence)
        self.train(is_training)
        return indices.round().type(self._dtype)

    def split(self, hidden, right_layer, joint_layer, existence):
        indices = self.diff_integer_indice(right_layer, joint_layer, existence)
        indices = torch.cumsum(indices, dim = 1)

        lhs_helper, rhs_helper, phy_jnt = condense_splitter(right_layer, joint_layer, existence)
        lhs_indices = condense_left(indices, lhs_helper, skip_dump0 = False)
        rhs_indices = condense_left(indices, rhs_helper, skip_dump0 = False)
        lhs_indices[:, 0] = 0
        rhs_indices[:, 0] = 0
        lhs_hidden = hidden[:, lhs_indices]
        rhs_hidden = hidden[:, rhs_indices]
        return lhs_hidden, rhs_hidden, phy_jnt, lhs_indices + rhs_indices

from itertools import count
from tqdm import tqdm
from utils.shell_io import byte_style
def train_model(tiger_reader, index_cnn, min_epoch = 5):
    from utils.types import M_TRAIN, M_DEVEL, M_TEST, O_LFT, O_RGT
    binarization = {O_LFT: 0.5, O_RGT: 0.5}
    # Small batch sizes work the best!
    ts_size, ts_iter = tiger_reader.batch(M_TRAIN, 10, 3, binarization, sort_by_length = False)
    ds_size, ds_iter = tiger_reader.batch(M_DEVEL, 10, 3, binarization, sort_by_length = True)
    _s_size, _s_iter = tiger_reader.batch(M_TEST,  10, 3, binarization, sort_by_length = True)
    opt = torch.optim.Adam(index_cnn.parameters(), 0.005)#, weight_decay = 1e-1)
    epoch_cnt = 0
    last_loss = 100
    for epoch_cnt in count(1):
        with tqdm(desc = f'{epoch_cnt}-th epoch', total = ts_size) as qbar:
            for batch in ts_iter:
                opt.zero_grad()
                batch_size, batch_len = batch['token'].shape
                target_hat = index_cnn(batch['right'], batch['joint'], batch['existence'])
                # target_hat =  * target_hat
                error = (target_hat - batch['target']) ** 2
                loss = error.sum()
                # loss = loss + (torch.norm(index_cnn._cnn.weight, 1) + torch.norm(index_cnn._cnn.bias, 1)) * 1e-1
                qbar.desc = f'{epoch_cnt}-th epoch, loss/tok: ' + byte_style(f'{loss / batch["seq_len"].sum():.6f}', '2' if loss < last_loss else '3')
                loss.backward()
                opt.step()
                qbar.update(batch_size)
        
        total_errors = 0
        with tqdm(desc = f'Validating {epoch_cnt}-th epoch', total = ds_size + _s_size) as qbar:
            for batch_iter in (ds_iter, _s_iter):
                for batch in batch_iter:
                    batch_size, batch_len = batch['token'].shape
                    target_hat = index_cnn.diff_integer_indice(batch['right'], batch['joint'], batch['existence'])
                    # target_hat = batch['indice_mask'] * target_hat
                    error = target_hat != batch['target']
                    num_errors = error.sum()
                    total_errors += num_errors
                    qbar.update(batch_size)
                    if num_errors:
                        qbar.desc = byte_style(f'{total_errors}', '1') + ' errors found!'
                        # import pdb; pdb.set_trace()
            if total_errors == 0 and epoch_cnt >= min_epoch:
                qbar.desc = byte_style(f'We get a clean Indexing-CNN{index_cnn._repr}!', '2')
                break
            elif total_errors:
                qbar.desc = byte_style(f'{total_errors}', '1') + ' errors found!'
            else:
                qbar.desc = byte_style('No', '2') + ' errors found!'

def check_model(tiger_reader, index_cnn):
    from utils.types import M_DEVEL, M_TEST, O_LFT, O_RGT
    binarization = {O_LFT: 0.5, O_RGT: 0.5}
    ds_size, ds_iter = tiger_reader.batch(M_DEVEL, 10, 3, binarization, sort_by_length = True)
    _s_size, _s_iter = tiger_reader.batch(M_TEST,  10, 3, binarization, sort_by_length = True)
    total_errors = 0
    with tqdm(desc = 'Validating', total = ds_size + _s_size) as qbar:
        for batch_iter in (ds_iter, _s_iter):
            for batch in batch_iter:
                batch_size, _ = batch['token'].shape
                target_hat = index_cnn.diff_integer_indice(batch['right'], batch['joint'], batch['existence'])
                # target_hat = batch['indice_mask'] * target_hat
                error = target_hat != batch['target']
                num_errors = error.sum()
                total_errors += num_errors
                qbar.update(batch_size)
                if num_errors:
                    qbar.desc = byte_style(f'{total_errors}', '1') + ' errors found!'
        safe = total_errors == 0
        if safe:
            qbar.desc = byte_style(f'We have a clean Indexing-CNN{index_cnn._repr}!', '2')
        else:
            qbar.desc = byte_style(f'{total_errors}', '1') + ' errors found!'
    return safe