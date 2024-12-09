#!/usr/bin/bash

#!/usr/bin/bash

mlflow run . \
       -P steps=train_random_forest \
       -P hydra_options="modeling.random_forest.max_depth=15,20,50 \
                         modeling.random_forest.n_estimators=100,200,500 \
                         modeling.random_forest.min_samples_split=2,4,8 \
                         modeling.random_forest.min_samples_leaf=1,3,5 \
                         modeling.random_forest.max_features=0.3,0.5,0.7 \
                         hydra/launcher=basic -m"