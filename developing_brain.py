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

#************************************************************************************
#load datasets
data_all = pd.read_pickle('developing_brain/exprMatrix.pkl')
data_cortex = pd.read_pickle('developing_brain/exprMatrix_cortex.pkl')
metadata = pd.read_pickle('developing_brain/meta.pkl')
metadata_cortex = pd.read_pickle('developing_brain/meta_cortex.pkl')
#************************************************************************************
plt.figure(figsize=(6,4))
sns.countplot(x = 'Structure', data=metadata)
plt.figure(figsize=(6,4))
sns.countplot(x = 'Area.fixed', data=metadata_cortex)
plt.figure(figsize=(8,4))
sns.countplot(x = 'Individual', data=metadata_cortex)
#************************************************************************************
#initiate the analysis module. now process the cortex only data. later this can be 
#changed to using all data, e.g. scMSAnalysis(metadata) and sc_db.preprocessing(data_all)
sc_db = scMSAnalysis(metadata_cortex)
#performing preprocessing, this case no normalization is performed. 
sc_db.preprocessing(data_cortex,norm_method='None',feat_drop_rate=0.005,cell_drop_rate=0.005)
#************************************************************************************
#get the labels for the cells.
sc_db.get_labels(['Age.fixed','Area.fixed','Individual'])
sc_db.metadata
#************************************************************************************
#perform analysis using the scanpy. this includes pca, umap, and leiden
#clustering, as well as differential analysis for each label.
sc_db.analyze(n_neighbors=15, n_pcs=15, min_dist=0.5, resolution = 0.15,
              categories=['Area.fixed','Age.fixed'])
#************************************************************************************
#interactive module that shows the cell embeddings.
show_embed = widgets.interactive(sc_db.show_cellEmbed, label=sc_db.adata.obs.columns,
                                 embed_method=sc_db.adata.obsm.keys(),size=(0.01,0.1,0.05));

show_embed
#************************************************************************************
#interactive module that shows the cell embeddings for each individual label from the above.
show_embed_label = widgets.interactive(sc_db.show_cellEmbed_label,label=show_embed.kwargs['label'],
                                 classes=sc_db.label_class[show_embed.kwargs['label']],
                                 embed_method=sc_db.adata.obsm.keys(),size=(0.01,0.1,0.05));

show_embed_label
#************************************************************************************
#interactive module that displays the ranked features for each label class.
label = 'Area.fixed' #change this to different label class.
show_feat = widgets.interactive(sc_db.show_featrank, label = label, region = sc_db.label_class[label], 
                                rank_by=sc_db.feat_rank[label].keys(), num_select=500,
                                num_show=IntSlider(min=0, max=500, step=20, value=0));
show_feat
#************************************************************************************



