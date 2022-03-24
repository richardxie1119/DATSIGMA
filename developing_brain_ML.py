import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('scms_py')
from scMSData import scMSData
from analysis import scMSAnalysis
from model import scMSModel

import ipywidgets as widgets
from ipywidgets import Box, IntSlider
%matplotlib widget
import random

#************************************************************************************
#load datasets
data_all = pd.read_pickle('developing_brain/exprMatrix.pkl')
data_cortex = pd.read_pickle('developing_brain/exprMatrix_cortex.pkl')
metadata = pd.read_pickle('developing_brain/meta.pkl')
metadata_cortex = pd.read_pickle('developing_brain/meta_cortex.pkl')
#************************************************************************************
#randomly sample 20% of cells for ML classification.
names = list(data_cortex.index)
name_sampled = random.sample(names,int(len(names)*0.2))
#************************************************************************************
#initiate the model class
scmodel = scMSModel(data_cortex.loc[name_sampled],
                    metadata_cortex.loc[name_sampled])

scmodel.get_labels(['Area.fixed','Age.fixed']) #encode numerical labels for the metadata labels
#************************************************************************************
#train a list of models with 5-fold cross validation
scmodel.train_models(cv=True,k=5, model_names=['DNN','GBT','RF','LR','SVM','LDA'], label_name='Area.fixed',
                  kwargs={'layer_shapes':[scmodel.intens_mtx.shape[1],64,32,6]})
#************************************************************************************
#analyze the test metrics for the cross validations.
f1 = []
model_names = []
for key in scmodel.test_metrics.keys():
    f1+=[metric['f1'] for metric in scmodel.test_metrics[key]]
    model_names+=[key for metric in scmodel.test_metrics[key]]
cv_df = pd.DataFrame({'f1':f1,'model':model_names})


plt.figure()
ax = sns.boxplot(x='model',y='f1',data=cv_df,fliersize=0, whis = 100, linewidth=1.3)
sns.stripplot(x='model',y='f1',data=cv_df,
              size=6, color=".3", linewidth=0)
for i,box in enumerate(ax.artists):
    box.set_edgecolor('black')
    box.set_facecolor('white')
plt.show()
#plt.savefig('figures/developing_brain_f1_10fold.pdf')