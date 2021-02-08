# UniTP
Unified Tokenization and Parsing framework in PyTorch

![NCCP](000/figures/nccp.gif)

## Requirements
- `pip install -r requirements/visual.txt` to visualize remote tensors locally through sftp. (funny!)
- `pip install -r requirements/full.txt` to train or test our models with PyTorch.
  - [Evalb](https://nlp.cs.nyu.edu/evalb/) is necessary.
  - [FastText](https://fasttext.cc/), [huggingface transformers](https://github.com/huggingface/transformers) and, [Discontinuous DOP](https://github.com/andreasvc/disco-dop) are optional.
  (recommmend for reproduce the results.)

## Models
- NCCP: Neural Combinatory Constituency Parsing (continuous)
  - ACCP: Attentional Combinatory Constituency Parsing (currently continuous)
- DCCP: Discontinuous Combinatory Constituency Parsing

## Usage

### Visualization
Once you `git clone https://github.com/this/git_repo` from this repository, the folder `000` contains
our best pre-trained BiLSTM models and visualization samples.

- Use `./visualization.py '000/lstm_nccp/0.ptb/penn_devel'` to see English continuous parsing training process.
- Check `'000/lstm_accp/0.ptb/penn_devel/*.art'` text files for multi-branching tree visualization with headedness.
(We use freely available Penn Treebank/PTB sections for this visualization.)
- Use `./visualization.py '000/lstm_dccp/0.dptb/disco_devel'` to see Engish discontinuous parsing training process.
- Use `./visualization.py '000/lstm_dccp/1.tiger/disco_devel'` to see German discontinuous parsing training process.

You can also train a new model with your corpus to see more details.

### Test a Pre-Trained Models
First, try training a new model with the pre-set configuration `000/manager.yaml`.
- Set `data:ptb:source_path:` to your PTB/WSJ folder, and then use `./manager.py 000 -p` to prepare the data.
- Set items under `tool:evalb:` for testing F1 scores.
  - Use `./manager.py 000` to check the status. Any improper configuration will be prompted.
- Use `./manager.py 000 -s lstm_nccp -i0` to test the pre-train model on PTB test set.
  - Add `-g [GPU ID]` to choose a GPU; the default is 0.
  - Use `./visualization.py '000/lstm_nccp/0.ptb/penn_test'` to see local tensors or add ` -h [server address] -p [port]` to see remote ones.

### Train a New Model
If you want a new work folder, try `mkdir Oh-My-Folder; ./manager.py Oh-My-Folder`. You will get a new configure file `Oh-My-Folder/manager.yaml`.
- Use `./manager.py Oh-My-Folder` to check the status and available experiments.
- Use `./manager.py Oh-My-Folder -s lstm_nccp/ptb:Oh-My-Model` to train a continuous model on PTB.
  - Add `-x [fv=fine evaluation start:[early stop count:[fine eval count]]],[max=max epoch count],[! test along with evaluation]` to change training settings.
  - Use `-s [lstm_nccp/[ptb/ctb/ktb]|xlnet_nccp]` to choose a continuous binary parsing experiment.
  - Use `-s [lstm_accp/[ptb/ctb/ktb]|xlnet_accp]` to choose a continuous multi-branching parsing experiment.
  - Use `-s [lstm_dccp|xbert_dccp]/[dptb/tiger]` to choose a discontinuous parsing experiment.
  - Use `-s [lstm_sentiment|xlnet_sentiment]` to run a joint task with [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/treebank.html). Set `task/lstm_sentiment/model/hidden_dim: null` as so to turn off the joint task. You can also check the SST tensors with `Oh-My-Folder/lstm_sentiment/0.Oh-My-Model/stan_devel` in a remote/local folder or similar test folders.
  - Use `-s [lstm_tokenization]/[ptb/ctb/ktb]` run a BPE-style neural tokenization. Also please try visualization:)

### Tips
- Try modifying the hyper-parameters in your `Oh-My-Folder/manager.yaml`.
- `Oh-My-Folder/lstm_nccp/register_and_tests.yaml` contains the scores for this experiment.
- Our reported top speed of NCCP is 1.3k sents/sec on GeForce GTX 1080 Ti with `task:[select]:train:multiprocessing_decode: true`. The speed is shown during the training process. To test the speed on the test set. Use `-x !` to include test set with evaluation. Otherwise, the top speed is around 0.5k sents/sec.

### Tips for Developers
- All modules in `experiments` prefixed with `t_` will be recognized by `manager.py`.
  - Feel free to write any model in your own experiment. For example, `t_lstm_tokenization` is 
  a young and underdeveloped experiment. We wrote most codes in the local `model.py` rather than in
  `models/`.
  - `models/` contains our sophisticated models' base classes such as `nccp.py`, `accp.py`, and `dccp.py`.
  If you are looking for our implementation, these three files plus `combine.py` are what you are looking for.
This framework is mainly for research. We do not pack vocabulary to the model. 