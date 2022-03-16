import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import ranksums
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.metrics import classification_report,confusion_matrix 
import SCCML
from utils import *
import os
from adjustText import adjust_text



#------------------------------------------------
# load the dataset
#------------------------------------------------
intens_df = loadmatfile('./data/vesicle_combined_aligned.mat')

rh_files = [file[:-4] for file in os.listdir('./data/Red Hemiduct FTICR Data')]
types = []
for name in intens_df.index:
    if name in rh_files:
        types.append('red hemiduct vesicle')
    else:
        types.append('dense core vesicle')
data_info = pd.DataFrame({'types':types})
data_info.index = intens_df.index

#------------------------------------------------
# apply filters to select features with cell number
#------------------------------------------------
data_filtered = intens_df.iloc[intens_df.astype(bool).sum(axis=1).values>20,intens_df.astype(bool).sum(axis=0).values>30]
data_filtered = data_filtered.iloc[:,(data_filtered.columns>500)&(data_filtered.columns<1100)]

print('filtered intensity matrix with shape {}'.format(data_filtered.shape))

#------------------------------------------------
# normalize (RMS)
#------------------------------------------------
norm_factors = np.sqrt(np.mean(data_filtered.replace(0,np.NaN)**2,axis=1)).values
norm_factors = norm_factors.reshape(data_filtered.shape[0],1)
data_filtered = np.divide(data_filtered,norm_factors)
features = data_filtered.columns.values

#------------------------------------------------
# t-SNE
#------------------------------------------------
X_embedded = TSNE(n_components=2,metric='cosine',perplexity=30,random_state=19,n_jobs=8).fit_transform(data_filtered)

pca_df = pd.DataFrame(X_embedded[:,:2],columns=['t-SNE1','t-SNE2'])
fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.grid(False)
fig = sns.scatterplot(x=pca_df['t-SNE1'], y=pca_df['t-SNE2'], s=12,alpha=0.7,hue=types,palette=['gray','lightcoral'])
plt.legend(frameon=False,fontsize=12,ncol=1,loc='lower center',bbox_to_anchor=(0.5, 1))

#------------------------------------------------
# 3-fold validation
#------------------------------------------------
fpr = []
tpr = []
AUC = []
pre = []
rec = []
acc = []
shap = []
CF = []
k= 3 #3-fold
sample_names = data_filtered.index.values
sample_idx_shuffled = shuffle(np.arange(0,sample_names.shape[0]),random_state=19)
sample_idx_kfold_test = np.array_split(sample_idx_shuffled,k)

for i in range(k):
    print('{} fold test..'.format(i))
    sample_idx_train = list(set(sample_idx_shuffled) - set(sample_idx_kfold_test[i]))
    sample_names_test = sample_names[sample_idx_kfold_test[i]]
    sample_names_train = sample_names[sample_idx_train]
    
    data_test = data_filtered.loc[sample_names_test]
    data_train = data_filtered.loc[sample_names_train]
    y_test = data_info.loc[sample_names_test]['types']
    y_train = data_info.loc[sample_names_train]['types']

    model = xgboost.XGBClassifier(n_estimators=500,random_state=19)

    model.fit(data_train.values,y_train)
    y_pred = model.predict(data_test.values)

    y_pred_prob = model.predict_proba(data_test.values)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    acc.append(report_dict['accuracy'])
    y_test_ = y_test.copy()
    y_test_[y_test_=='dense core vesicle']= int(1)
    y_test_[y_test_=='red hemiduct vesicle']= int(0)
    y_test_ = y_test_.astype(int)

    CF.append(confusion_matrix(y_test,y_pred))
    f,t,a = roc_curve(y_test_,y_pred_prob[:,0],drop_intermediate=False)
    p,r,a2 = precision_recall_curve(y_test_,y_pred_prob[:,0])
    contrib_xgb_best = SCCML.feature_contrib(model,data_filtered,data_filtered.columns.values,10,False,False)
    shap.append(contrib_xgb_best)
    fpr.append(f);tpr.append(t);AUC.append(auc(f,t));pre.append(p);rec.append(r)


print('3 fold test accuracy using all features: {}+={}'.format(np.mean(acc),np.std(acc)))

#------------------------------------------------
# plot confusion matrix for test data in each fold
#------------------------------------------------
plot_confusion_matrix(CF[0], classes=['dense core vesicle','red hemiduct vesicle'])
plot_confusion_matrix(CF[1], classes=['dense core vesicle','red hemiduct vesicle'],)
plot_confusion_matrix(CF[2], classes=['dense core vesicle','red hemiduct vesicle'],)

#------------------------------------------------
# Compute and rank the features using mean abs SHAP
#------------------------------------------------
shap_ranked_index = np.argsort(abs(contrib_xgb_best).mean(0))[::-1]

X_df = data_filtered.copy()

contrib_df = pd.DataFrame()
contrib_df['m/z'] = features[shap_ranked_index]
contrib_df['mean SHAP'] = np.around(abs(contrib_xgb_best).mean(0),4)[shap_ranked_index]
contrib_df['mean SHAP'] = np.round(contrib_df['mean SHAP']/contrib_df['mean SHAP'].max(),4)
a=X_df.values
ratio = a[data_info['types']=='red hemiduct vesicle'].mean(0)/a[data_info['types']=='dense core vesicle'].mean(0)
contrib_df.loc[ratio[shap_ranked_index]>1,'contribute to which GOI']='red hemiduct vesicle'
contrib_df.loc[ratio[shap_ranked_index]<=1,'contribute to which GOI']='dense core vesicle'
contrib_df = contrib_df[contrib_df['mean SHAP']!=0]
contrib_df.head(50)

#------------------------------------------------
# plot the summary plot of most contributing features
# for the classification task 
#------------------------------------------------
mean_intens = []
for row in contrib_df.iterrows():
    mean_intens.append(data_filtered[data_info['types']==row[1]['contribute to which GOI']][row[1]['m/z']].mean())
pvals=[]
for mz in contrib_df['m/z']:
    pvals.append(ranksums(data_filtered[np.array(types)=='dense core vesicle'][mz],data_filtered[np.array(types)=='red hemiduct vesicle'][mz])[1])
contrib_df['-log10(p values)'] = -np.log10(np.array(pvals))

plt.figure(figsize=(4,4))
g = sns.JointGrid(height=4, ratio=5, space=.05,marginal_ticks=True)
gs=sns.scatterplot(x='m/z', y='mean SHAP',size='-log10(p values)',alpha=0.7,hue='contribute to which GOI',palette=['gray','lightcoral'],data=contrib_df,marker='o',ax=g.ax_joint)
sns.histplot(x='m/z', linewidth=1.1,data=contrib_df, ax=g.ax_marg_x,bins=20,alpha=0.5,color='black',fill=False)
sns.histplot(y='mean SHAP', linewidth=1.1,data=contrib_df, ax=g.ax_marg_y,alpha=0.5,color='black',fill=False)
#plt.legend(frameon=False,fontsize=12,ncol=1,loc=2,bbox_to_anchor=(1.05, 1),borderaxespad=0)
gs.legend(bbox_to_anchor=(1.2, 1), borderaxespad=0.,frameon=False)

#------------------------------------------------
# volcano plot
#------------------------------------------------
data_filtered_selected = data_filtered[features[shap_ranked_index[:97]]]
treated_data = data_filtered_selected[data_info['types']=='dense core vesicle']
nontreated_data = data_filtered_selected[data_info['types']=='red hemiduct vesicle']

treated_mean = treated_data.mean(0)
nontreated_mean = nontreated_data.mean(0)
log2fold = np.log2(treated_mean/nontreated_mean).values
log10pvals = np.array([-np.log10(ranksums(treated_data[mz], nontreated_data[mz])[1]) for mz in features[shap_ranked_index[:97]]])


fig,axes = plt.subplots(1,1,figsize=(5,4))
axes.scatter(log2fold, log10pvals,s=14,c='gray')
axes.axvline(0,linestyle='--',c='k',alpha=0.5)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
plt.xlabel('log2(treated/nontreated)')
plt.ylabel('-log10(p-values)')
annot_idx = 0

