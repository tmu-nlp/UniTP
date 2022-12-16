# UniTP
Unified Tokenization and Parsing framework (UniTP) in PyTorch for our two papers in [ACL Findings 2021](https://aclanthology.org/2021.findings-acl.194) and 
[TACL 2022](https://aclanthology.org/2022.tacl-na.na).
This is Neural Combinatory Constituency Parsing (NCCP) family which also performs addtional word segmentation (WS), sentiment analysis (SA), named entity recoginition (NER).

This project is extended from [https://github.com/tmu-nlp/nccp](https://github.com/tmu-nlp/nccp).

![NCCP](000/figures/nccp.gif)

## Requirements

For models with fastText,
- `pip install -r requirements/minimal.txt`
- Install [fastText](https://fasttext.cc/) and configurate values under path `tool:fasttext:` in file `000/manager.yaml`.

Additional requirements:
- `pip install -r requirements/full.txt` for NCCP models with [huggingface transformers](https://github.com/huggingface/transformers).
- For continuous models, install [evalb](https://nlp.cs.nyu.edu/evalb/) and configurate values under path `tool:evalb:` in file `000/manager.yaml`.
- For discontinuous models, install [discontinuous DOP](https://github.com/andreasvc/disco-dop) and configurate values under path `tool:evalb_lcfrs_prm:` in file `000/manager.yaml`.

## Neural Combinatory Constituency Parsing Models
- CB: continuous and binary (`models/nccp`, +SA)
- CM: continuous and multi-branching (`models/accp`, +WS, +NER)
- DB: discontinuous binary (`models/dccp`)
- DM: discontinuous multi-branching (`models/xccp`)

Besides constituency parsing, continuous models enables SA, WS, and NER.
All models can be either monolingual or multilingual.

## Usage
### Train a monolingual model
We provide configuration of models in our two papers
(i.e., [CB and CM](https://aclanthology.org/2021.findings-acl.194) as file `000/manager.yaml` and [DB and DM](https://aclanthology.org/2022.tacl-na.na)).
Please first configurate path `data:[corpus]:source_path:` for each corpus you have and check additional requirements above.

If a monolingual parser as in our published papers, you might try the following command:
    
    # train DB on corpus DPTB on device GPU ID 0.
    ./manager.py 000 -s db/dptb -g 0

    # give a optional folder name [#.test_me] for storage
    ./manager.py 000 -s db/dptb:test_me -g 0

For multiligual models, example are:

    # train CB on all available corpora (i.e., PTB, CTB, KTB, NPCMJ, and SST (for SA) if configurated) on GPU 0 (using default values).
    ./manager.py 000 -s cb

    # train CM on all available corpora (i.e., PTB, CTB, KTB, NPCMJ, and CONLL & IDNER (for NER) if configurated).
    ./manager.py 000 -s cm

    # train DM on corpora DPTB and TIGER with pre-trained language models on device GPU 4.
    ./manager.py 000 -s pre_dm/dptb,tiger -g 4
    # or
    ./manager.py 000 -s pre_dm -g 4

### Test a trained model

Each trained model is stored at `000/[model]/[#.folder_name]` with an entry `[#]` in `000/[model]/register_and_tests.yaml`, where `[model]` is the model variant and `[#]` is an integer for the trained model instance. Number `[#]` is assigned by `mamager.py` and `[folder_name]` is the optional name for storage such as `:test_me` in the previous example.

To test, you may try:

    ./manager 000 -s [model] -i [#]

### Hyperparameter tuning

We suggest tuning hyperparameter with a trained model.

    ./manager 000 -s [model] -ir [#] -x mp,optuna=[#trials],max=0

If you want to edit the range of hyperparameter explorating, please find a respective file `experiments/[model]/operator.py` and modify its function `_get_optuna_fn`.

### Visualization (not avaliable now)
We gave an exemplary illustration from our project.
However, because there are no much demand for visualization, `./visualization.py` becomes obsolated.