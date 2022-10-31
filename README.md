
Implementation for paper:

Multi-granularity Optimization for Non-autoregressive Translation (To appear in EMNLP 2022)

[Paper Link](https://arxiv.org/abs/2210.11017)

<!-- ## Replication   -->

## Setup
```
pip install -e . 
pip install tensorflow tensorboard sacremoses nltk Ninja omegaconf
pip install 'fuzzywuzzy[speedup]'
pip install hydra-core==1.0.6
pip install sacrebleu==1.5.1
pip install git+https://github.com/dugu9sword/lunanlp.git
```

## Experimental Details
### Hyperparameters
#### Pretrain stage
|                             	| EN<->RO 	| EN<->DE 	|
|-----------------------------	|---------	|---------	|
| --validate-interval-updates 	| 300     	| 300     	|
| number of tokens per batch  	| 32K     	| 128K    	|
| --dropout                   	| 0.3     	| 0.1     	|
| --max-update                   	| 300k     	| 300k     	|
#### MgMO stage
|                             	| EN<->RO 	| EN<->DE 	|
|-----------------------------	|---------	|---------	|
| --validate-interval-updates 	| 300     	| 300     	|
| number of tokens per batch  	| 256     	| 1024    	|
| --dropout                   	| 0.1     	| 0.1     	|
| --lr-scheduler                   	| fixed    	| fixed     	|
| --lr             | 2e-6 | 2e-6 |
### Arguments
|Argument	|Description	|
|---------------	|------------------------------------------------	|
| --n-sample 	| Number of samples in the search space      	|
| --reward-alpha  	| Coefficient for balancing the sentence probability     	|
| --max-length-bias         	| Max deviation of the predicted length to the golden length during training    	|
| --null-input | Set for N&P mode (default) |
| --rm-scale | The gamma for controling the granularity size |
| --len-loss | Set to enable length loss during training |


### Training
We provide a script (run.sh) for replicating the results on WMT'16 EN->RO task. For other directions, you need to adjust the data path and corresponding hyper-paramters where necessary.




### Evaluation
We select the best checkpoint for evaluation based on the validation BLEU scores. We set the length beam as 5 for inference. See `run.sh' for details.


## Main Files

The implementation is based on Fairseq. We mainly add the following files.


```
fairseq
├── criterions
│   └── multi_granularity_optimizer.py  # mutli-granularity loss
└── models
    └── nat
        └── cmlm_transformer.py         # implementation for sampling and granularity generation

```
