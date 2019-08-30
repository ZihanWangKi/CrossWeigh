# CrossWeigh

## Data
`/data/corrected.testb.iobes` folder is the manually corrected test set, there should be exactly 186 sentences that 
differ from the original test set.

## Scripts
`split.py` can be used to generate a k-fold entitiy disjoint dataset from a list of datasets(usually both the train and development set)  
`flair_scripts/flair_ner.py` can be used to train a weighted version of flair.  
`collect.py` can be used to collect all the predictions on the k folded test set.  