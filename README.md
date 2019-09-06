# CrossWeigh
<h1 align="center">CrossWeigh</h1>
<h5 align="center">CrossWeigh: Training Named Entity Tagger from Imperfect Annotations</h5>

## Motivation 

The label annotation mistakes by human annotators brings up two challenges to NER:  
- mistakes in the test set can interfere the evaluation results and even lead to an inaccurate assessment of model performance.
- mistakes in the training set can hurt NER model training. 

We address these two problems by:
- manually correcting the mistakes in the test set to form a cleaner benchmark.
- develop framework `CrossWeigh` to handle the mistakes in the training set. 
<embed src="img/CrossWeigh.pdf" width="800px" height="500px" />

`CrossWeigh` works with any NER algorithm that accepts weighted training instances. It
is composed of two modules. 1) mistake estimation: where potential mistakes are identified in the training
data through a cross-checking process and 2) mistake re-weighing: where weights of those mistakes are lowered
during training the final NER model.

## Data
`/data/corrected.testb.iobes` folder is the manually corrected test set, there should be exactly 186 sentences that 
differ from the original test set.

## Scripts
`split.py` can be used to generate a k-fold entity disjoint dataset from a list of datasets(usually both the train and development set)  
`flair_scripts/flair_ner.py` can be used to train a weighted version of flair.  
`collect.py` can be used to collect all the predictions on the k folded test set.  

## Steps to reproduce
Make sure you are in a python3.6+ environment.  
See [example.sh](example.sh) to reproduce the results.  
Using [Flair](https://github.com/zalandoresearch/flair) (non-pooled version), the final result should achieve
around 93.19F1 on the original test dataset and 94.18F1 on the corrected test set. Using Flair without CrossWeigh gives
around 92.9F1.  

## Citation
Please cite the following paper if you found our dataset or framework useful. Thanks!

>Zihan Wang, Jingbo Shang, Liyuan Liu, Lihao Lu, Jiacheng Liu, and Jiawei Han. "CrossWeigh: Training Named Entity Tagger from Imperfect Annotations." arXiv preprint arXiv:1909.01441 (2019).

```
@article{wang2019cross,
  title={CrossWeigh: Training Named Entity Tagger from Imperfect Annotations},
  author={Wang, Zihan and Shang, Jingbo and Liu, Liyuan and Lu, Lihao and Liu, Jiacheng and Han, Jiawei},
  journal={arXiv preprint arXiv:1909.01441},
  year={2019}
}
```
