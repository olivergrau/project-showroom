#!/usr/bin/bash

mlflow run . -P steps=download,basic_cleaning,data_split,train_random_forest