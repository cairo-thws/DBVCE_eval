# (c) Philipp Väth and Alexander Frühwald 2023
# This script generates the pandas tables for the paper.

import pandas as pd 
import glob
import numpy as np

# read all csv files in ./csv
df = pd.concat([pd.read_csv(f) for f in glob.glob('./analysis/csv/*.csv')], ignore_index = True)	



### DATA 1
dy = df.copy()
# filter for [source_class, target_class] combinations
valid_combinations = [
    (293, 288),
    (209, 207),
    (935, 924),
    (973, 970),
    (970, 980),
    (965, 963),
]

dy = dy[dy[['source_class', 'target_class']].apply(tuple, axis=1).isin(valid_combinations)]

dy = dy.set_index(['classifier', "source_class_name", "target_class_name", "cone_projection", "xzero_prediction",])

# remove some columns
dy = dy.drop(columns=["source_class", "target_class", "run_id"])
#dy = dy.drop(columns=["source_class_name", "target_class_name", "run_id"])



#%%
dy = dy.reset_index()
dy = dy[dy["cone_projection"] == True ]
dy = dy[dy["xzero_prediction"] == True ]
dy = dy.drop(columns=["cone_projection", "xzero_prediction"])

#dy = dy.set_index(['classifier', "source_class_name", "target_class_name", "cone_projection", "xzero_prediction",])

dy = dy.groupby(['classifier'])

#print(dy.mean().head(30))

def mean_std(data):
    arr = np.array(data)
    std, mean = np.around(arr.std(),decimals=2), np.around(arr.mean(), decimals=2)
    
    if mean >= 100:
        std, mean = int(std), int(mean)
    
    #if std != 0.0:
    return f"{mean} ± {std}"
    #else:
    #    return f"{mean}"
    
dy = dy.agg(mean_std)


#print(dy.head(30))
#dy = dy.agg(['mean', 'std'])

# Cosmetics
#dy = dy.round({"target_class_conf_std":2, "target_class_conf_mean":2,"original_class_conf_std":2, "original_class_conf_mean":2, "fid":2,"target_class_validity": 2, "original_class_validity": 2, "lp_norm_1" : 0, "lp_norm_1_5" : 0, "lp_norm_2": 0, "lpips": 2, "oracle_MNR-RN50":2, "oracle_AlexNet (pretrained)":2, "oracle_AlexNet (random init)":2, "oracle_UnetEncoder":2, "oracle_ConvNeXt-L":2, "oracle_SIMCLR":2, "oracle_SwinTFL":2 })
#dy = dy.astype({'lp_norm_1': 'int32', 'lp_norm_1_5': 'int32', 'lp_norm_2': 'int32'})
dy = dy.rename(columns={"target_class_conf_std":"TConfStd ↓", "target_class_conf_mean":"TConf ↑","original_class_conf_std":"OConfStd ↓", "original_class_conf_mean":"OConf ↓", "fid":"FID ↓","target_class_validity": "TCV ↑", "original_class_validity": "OCV ↓", "lp_norm_1" : "L1 ↓", "lp_norm_1_5" : "L1.5 ↓", "lp_norm_2": "L2 ↓", "lpips": "LPIPS ↓", "oracle_MNR-RN50":"OS Madry ↑", "oracle_AlexNet (pretrained)":"OS Alex ↑", "oracle_AlexNet (random init)":"OS AlexRand ↓", "oracle_UnetEncoder":"OS Unet ↑", "oracle_ConvNeXt-L":"OS ConvNeXt ↑", "oracle_SIMCLR":"OS SIMCLR ↑", "oracle_SwinTFL":"OS SwinTF ↑" })


def highlight(col):
    s = col.str.split(' ± ')
    s_mean_value = s.str[0].astype('float')
    if '↓' in col.name:
        col[s_mean_value.argmin()] = f"\textbf{{{col[s_mean_value.argmin()]}}}"
    elif '↑' in col.name:
        col[s_mean_value.argmax()] = f"\textbf{{{col[s_mean_value.argmax()]}}}"
    return col

dy = dy.apply(highlight,axis=0)

#print(dy.head(30))

dy = dy.T
print(dy.head(30))
dy.to_latex("analysis/out_mean.tex", escape=False)


#%% 
# import seaborn as sns
# import matplotlib.pyplot as plt

# f, ax = plt.subplots(figsize=(9, 6))
# hm = sns.heatmap(dy, annot=True, fmt=".3f", ax=ax)
# hm.set_title("Heatmap of mean values")
# plt.savefig("heatmap.png")


### Data 2
dz = df.copy()
dz = dz[dz["classifier"] == "MadryClassifier"]
dz = dz[dz["cone_projection"] == True ]
dz = dz[dz["xzero_prediction"] == True ]

dz = dz.drop(columns=["classifier", "source_class", "target_class", "run_id","cone_projection","xzero_prediction"])
dz = dz.set_index(["source_class_name", "target_class_name"])

dz = dz.round({"target_class_conf_std":2, "target_class_conf_mean":2,"original_class_conf_std":2, "original_class_conf_mean":2, "fid":2,"target_class_validity": 2, "original_class_validity": 2, "lp_norm_1" : 0, "lp_norm_1_5" : 0, "lp_norm_2": 0, "lpips": 2, "oracle_MNR-RN50":2, "oracle_AlexNet (pretrained)":2, "oracle_AlexNet (random init)":2, "oracle_UnetEncoder":2, "oracle_ConvNeXt-L":2, "oracle_SIMCLR":2, "oracle_SwinTFL":2 })
dz = dz.astype({'lp_norm_1': 'int32', 'lp_norm_1_5': 'int32', 'lp_norm_2': 'int32'})
dz = dz.rename(columns={"target_class_conf_std":"TConf_Std ↓", "target_class_conf_mean":"TConf ↑","original_class_conf_std":"OConf_Std ↓", "original_class_conf_mean":"OConf ↓", "fid":"FID ↓","target_class_validity": "TCV ↑", "original_class_validity": "OCV ↓", "lp_norm_1" : "L1 ↓", "lp_norm_1_5" : "L1.5 ↓", "lp_norm_2": "L2 ↓", "lpips": "LPIPS ↓", "oracle_MNR-RN50":"OS_Madry ↑", "oracle_AlexNet (pretrained)":"OS_Alex ↑", "oracle_AlexNet (random init)":"OS_AlexRand ↓", "oracle_UnetEncoder":"OS_Unet ↑", "oracle_ConvNeXt-L":"OS_ConvNeXt ↑", "oracle_SIMCLR":"OS_SIMCLR ↑", "oracle_SwinTFL":"OS_SwinTF ↑" })

print(dz.head(30))
