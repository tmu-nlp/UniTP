from torch import nn, cat, no_grad, int64, optim

def interweave(right_layer, joint_layer):
    batch_size, batch_len = joint_layer.shape
    r0 = right_layer[:, 0, None]
    jc = joint_layer[:, :, None]
    j1 = right_layer[:, 1:, None]
    jrjr = cat([jc, j1], dim = 2)
    jrjr = jrjr.reshape(batch_size, batch_len << 1)
    rjrjr = cat([r0, jrjr], dim = 1)
    return rjrjr.unsqueeze(dim = 1)

class Index(nn.Module):
    def __init__(self, dtype = int64):
        super().__init__()
        self._dtype = int64
        # self._dp = nn.Dropout(0.001)
        self._cnn = nn.Conv1d(1, 3, 9, stride = 2, padding = 4) #, bias = False)
        self._act = nn.Tanh()

    def forward(self, right_layer, joint_layer):
        channels = self._cnn(interweave(right_layer, joint_layer))
        return self._act(channels).sum(dim = 1)

    def integer_indice(self, right_layer, joint_layer):
        with no_grad():
            indices = self.forward(right_layer, joint_layer)
            indices = indices.round().type(self._dtype)
        return indices

def train_model(save_dir): # should be an operator function / model function from the reader/data
    from data.tiger import TigerReader
    from utils.types import M_TEST, O_LFT, O_RGT
    tiger_reader = TigerReader(save_dir, None, True, train_indexing_cnn = True)
    ds_size, ds_iter = tiger_reader.batch(M_TEST, 10, 3, {O_LFT: 0.9, O_RGT: 0.1})
    for batch in ds_iter:
        import pdb; pdb.set_trace()
    m = Index()
    opt = optim.Adam(m.parameters(), 0.01)#, weight_decay = 3e-1)

    for i in range(10000):
        y_hat = m(ojo)
        # print(y_hat.shape, y.shape)
        err = (y - y_hat.sum(dim = 1)) ** 2
        loss = err.sum()
        loss = loss + (torch.norm(m._l1.weight, 1) + torch.norm(m._l1.bias, 1)) * 3e-1
        if i % 500 == 499:
            print(loss)
        loss.backward()
        opt.step()
        opt.zero_grad()

    y_int = y.type(torch.int8)
    y_hat_sum = y_hat.sum(dim = 1)
    y_hat_int = y_hat_sum.round().type(torch.int8)