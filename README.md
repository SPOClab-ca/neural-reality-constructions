# Neural reality of argument structure constructions

This repository contains the source code and data for our ACL 2022 paper: *"Neural reality of argument structure constructions"* by Bai Li, Zining Zhu, Guillaume Thomas, Frank Rudzicz, and Yang Xu.

## Citation

If you use our work in your research, please cite:

Li, B., Zhu, Z., Thomas, G., Rudzicz, F., and Xu, Yang. 2022. Neural reality of argument structure constructions. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)*.

```
@inproceedings{li2022neuralreality,
  author = "Li, Bai and Zhu, Zining and Thomas, Guillaume and Rudzicz, Frank and Xu, Yang",
  title = "Neural reality of argument structure constructions",
  booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)",
  publisher = "Association for Computational Linguistics",
  year = "2022",
}
```

## Dependencies

The project was developed with the following library versions.

* Python 3.9.9
* numpy==1.22.0
* pandas==1.3.5
* scipy==1.7.3
* scikit-learn==1.0.2
* torch==1.10.1
* transformers==4.15.0

## Setup Instructions

1. Clone this repo: `git clone https://github.com/SPOClab-ca/neural-reality-constructions`
2. Download BNC Baby (4m word sample) from [this link](http://www.natcorp.ox.ac.uk/) and extract into `data/bnc/`
3. Run BNC preprocessing script: `python scripts/process_bnc.py --bnc_dir=data/bnc/download/Texts --to=data/bnc.pkl`
2. (Optional) Run unit tests: `PYTHONPATH=. python -m pytest test`

## Run sentence sorting (Case study 1)

```
PYTHONPATH=. python scripts/run_sentence_grouping.py --dataset=templates --model_name=roberta-base
```

Outputs four numbers: mean verb deviation, mean construction deviation, std of verb deviation, std of construction deviation.

## Run Jabberwocky (Case study 2)

```
PYTHONPATH=. python scripts/run_jabberwocky.py --condition high-freq
```
