# UniTP
Unified Tokenization and Parsing framework in PyTorch

![NCCP](000/figures/nccp.gif)

## Requirements
- `pip install -r requirements/visual.txt` to visualize remote tensors locally through sftp. (funny!)
- `pip install -r requirements/full.txt` to train or test our models with PyTorch.
  - [Evalb](https://nlp.cs.nyu.edu/evalb/) is necessary.
  - [FastText](https://fasttext.cc/) and [Discontinuous DOP](https://github.com/andreasvc/disco-dop) are optional.
  (recommmend for reproduce the results.)

## Models
- NCCP: Neural Combinatory Constituency Parsing (continuous)
- DCCP: Discontinuous Combinatory Constituency Parsing

## Usage

### Visualization
Once you `git clone https://github.com/tmu-nlp/UniTP` this repository, the folder `000` contains
our best pre-trained LSTM models and visualization samples.

- Use `./visualization.py '000/lstm_nccp/0.ptb/penn_devel'` to see English continuous parsing training process.
- Use `./visualization.py '000/lstm_dccp/1.dptb/disco_devel'` to see Engish discontinuous parsing training process.
- Use `./visualization.py '000/lstm_dccp/2.tiger/disco_devel'` to see German discontinuous parsing training process.
(We use freely available Penn Treebank/PTB sections for this visualization.)

You can also train a new model with your corpus to see more details.

### Test a Pre-Trained Models
First, try training a new model with the pre-set configuration `000/manager.yaml`.
- Set `data/ptb/source_path:` to your PTB/WSJ folder, and then use `./manager.py 000 -p` to prepare the data.
- Set items under `tool/evalb` for testing F1 scores.
  - Use `./manager.py 000` to check the status. Any improper configuration will be prompted.
- Use `./manager.py 000 -s lstm_nccp -i0` to test the pre-train model on PTB test set.
  - Add `-g [GPU ID]` to choose a GPU; the default is 0.
  - Use `./visualization.py '000/lstm_nccp/0.ptb/penn_test'` to see local tensors or add ` -h [server address] -p [port]` to see remote ones.

### Train a New Model
If you want a new work folder, try `mkdir Oh-My-Folder; ./manager.py Oh-My-Folder`. You will get a new configure file `Oh-My-Folder/manager.yaml`.
- Use `./manager.py Oh-My-Folder` to check the status and available experiments.
- Use `./manager.py Oh-My-Folder -s lstm_nccp/ptb:Oh-My-Model` to train a continuous model on PTB.
  - Add `-x [fv=fine evaluation start:[early stop count:[fine eval count]]],[max=max epoch count],[! test along with evaluation]` to change training settings.
  - Use `-s [lstm_nccp|xlnet_nccp]/[ptb/ctb/ktb]` to choose a continuous parsing experiment.
  - Use `-s [lstm_dccp|xbert_dccp]/[dptb/tiger]` to choose a discontinuous parsing experiment.
  - Use `-s [lstm_sentiment|xlnet_sentiment]` to run a joint task with [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/treebank.html). Set `task/lstm_sentiment/model/hidden_dim: null` as so to turn off the joint task. You can also check the SST tensors with `Oh-My-Folder/lstm_sentiment/0.Oh-My-Model/stan_devel` in a remote/local folder or similar test folders.
  - Use `-s [lstm_tokenization]/[ptb/ctb/ktb]` run a BPE-style neural tokenization. Also try visualization:)

### Tips
- Try modifying the hyper-parameters in your `Oh-My-Folder/manager.yaml`.
- `Oh-My-Folder/lstm_nccp/register_and_tests.yaml` contains the scores for this experiment.