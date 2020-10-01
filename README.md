# home-loans-deep-learning

## Getting source data

Download from Kaggle the Home Loans Default Risk dataset

```
! pip install kaggle
! export KAGGLE_USERNAME=<your_kaggle_username>
! export KAGGLE_KEY=<api_token_key>
! kaggle competitions download -c home-credit-default-risk -p data/kaggle
```

## Pre-processing

run:
`python src/process_data.py`

## Training
run:
`python src/train.py`

Model will be saved in `models` directory.

**IN PROGRESS**