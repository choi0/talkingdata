# References:
# https://www.kaggle.com/shujian/mlp-starter
# https://www.kaggle.com/titericz/giba-darragh-ftrl-rerevisited
# https://www.kaggle.com/joaopmpeinado/talkingdata-xgboost-lb-0-951
# https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data
# https://www.kaggle.com/cartographic/undersampler
# https://www.kaggle.com/prashantkikani/weighted-app-chanel-os
# https://www.kaggle.com/ogrellier/ftrl-in-chunck

LOGIT_WEIGHT = .8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.special import expit, logit

almost_zero = 1e-10
almost_one = 1 - almost_zero

models = {
  'xgb  ':  "../input/jo-o-s-xgboost-9610-version/xgb_sub.csv",
  'ftrl1':  "../input/giba-darragh-ftrl-rerevisited/sub_proba.csv",
  'nn   ':  "../input/shujian-s-mlp-starter-9502-version/sub_mlp.csv",
  'lgb  ':  "../input/pranav-s-lightgbm-9631-version/sub_lgb_balanced99.csv",
  'usam ':  "../input/lewis-undersampler-9562-version/pred.csv",
  'means':  "../input/weighted-app-chanel-os/subnew.csv",
  'ftrl2':  "../input/olivier-s-multi-process-ftrl-revised-9606-vesion/ftrl_submission.csv"
  }
  
weights = {
  'xgb  ':  .02,
  'ftrl1':  .09,
  'nn   ':  .04,
  'lgb  ':  .60,
  'usam ':  .05,
  'means':  .10,
  'ftrl2':  .10
  }
  
print (sum(weights.values()))


subs = {m:pd.read_csv(models[m]) for m in models}
first_model = list(models.keys())[0]
n = subs[first_model].shape[0]

ranks = {s:subs[s]['is_attributed'].rank()/n for s in subs}
logits = {s:subs[s]['is_attributed'].clip(almost_zero,almost_one).apply(logit) for s in subs}

logit_avg = 0
rank_avg = 0
for m in models:
    s = logits[m].std()
    print(m, s)
    logit_avg = logit_avg + weights[m]*logits[m] / s
    rank_avg = rank_avg + weights[m]*ranks[m]

logit_rank_avg = logit_avg.rank()/n
final_avg = LOGIT_WEIGHT*logit_rank_avg + (1-LOGIT_WEIGHT)*rank_avg

final_sub = pd.DataFrame()
final_sub['click_id'] = subs[first_model]['click_id']
final_sub['is_attributed'] = final_avg

final_sub.to_csv("sub_mix.csv", index=False)