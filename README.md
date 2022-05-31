# Classifying news from 20NewsGroup

[![Apache 2.0 licensed](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://github.com/albertobas/ames-housing-prices/blob/main/LICENSE)

## About

Supervised learning analysis on the [20NewsGroup](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups) dataset, which contains 20000 messages taken from 20 newsgroups.

There is an English and a Spanish version, both in Jupyter notebook format.

## Requirements

- python==3.9.7
- jupyter==1.0.0
- torch==1.10.1
- spacy==2.3.5
- matplotlib==3.5.0
- tqdm==4.62.3
- numpy==1.19.2
- scikit-learn==1.0.2

## Running locally

```bash
$ git clone https://github.com/albertobas/20-news-classification.git
$ cd 20-news-classification
$ conda env create --file environment.yml
$ conda activate 20_news_classification
$ python -m spacy download en_core_web_sm
$ jupyter notebook
```
