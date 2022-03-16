import os
from glob import glob
import sys

from processing import *
from tqdm import tqdm
import pandas as pd

from scipy.stats import spearmanr, ttest_ind
import seaborn as sns
import anndata
import scanpy as sc
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder



class scMSAnalysis():

    """
    """

    def __init__(self, metadata = None):

        self.metadata = metadata


    def preprocessing(self, intens_mtx, norm_method = None, feat_drop_rate = 0.05, cell_drop_rate = 0):

        """
        """

        intens_mtx = intens_mtx.iloc[:,intens_mtx.astype(bool).sum(axis=0).values>feat_drop_rate*intens_mtx.shape[0]]
        intens_mtx = intens_mtx.iloc[intens_mtx.astype(bool).sum(axis=1).values>cell_drop_rate*intens_mtx.shape[1],:]

        if norm_method == None:
            intens_mtx = intens_mtx

        if norm_method == 'l1':
            norm_factors = np.linalg.norm(intens_mtx,ord=1,axis=1)
            intens_mtx = intens_mtx/norm_factors.reshape(intens_mtx.shape[0],1)

        if norm_method == 'l2':
            norm_factors = np.linalg.norm(intens_mtx,ord=2,axis=1)
            intens_mtx = intens_mtx/norm_factors.reshape(intens_mtx.shape[0],1)

        if norm_method == 'max':
            norm_factors = intens_mtx.max(axis=1).values
            intens_mtx = intens_mtx/norm_factors.reshape(intens_mtx.shape[0],1)

        if norm_method =='mean':
            norm_factors = np.mean(intens_mtx.replace(0,np.NaN),axis=1).values
            intens_mtx = intens_mtx/norm_factors.reshape(intens_mtx.shape[0],1)

        if norm_method == 'rms':
            norm_factors = np.sqrt(np.mean(intens_mtx.replace(0,np.NaN)**2,axis=1)).values
            intens_mtx = intens_mtx/norm_factors.reshape(intens_mtx.shape[0],1)

        if norm_method == 'median':
            norm_factors = np.nanmedian(intens_mtx.replace(0,np.NaN),axis=1)
            intens_mtx = intens_mtx/norm_factors.reshape(intens_mtx.shape[0],1)


        self.intens_mtx = intens_mtx

        if type(self.intens_mtx.columns[0]) == float:
            self.intens_mtx.columns = np.round(self.intens_mtx.columns.astype(float),4).astype(str)


        self.metadata = self.metadata.loc[self.intens_mtx.index]

        self.adata = anndata.AnnData(self.intens_mtx)
        self.adata.var['mz'] = self.intens_mtx.columns.astype(str)

        print('filtered intensity matrix with shape {}'.format(self.intens_mtx.shape))


    def get_labels(self, labels):

        self.label_class = {}

        for label in labels:
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(self.metadata[label])

            self.label_class[label] = label_encoder.classes_
            self.metadata[label+'_int'] = integer_encoded

            if self.metadata[label].values.dtype == int:
                self.metadata[label] = self.metadata[label].astype(str)

            self.adata.obs[label] = self.metadata[label]


    def analyze(self, n_pcs, n_neighbors, min_dist, resolution, labels=None, categories=['leiden'], log=False):

        if log:
            sc.pp.log1p(self.adata)
            self.intens_mtx = np.log1p(self.intens_mtx)

        sc.tl.pca(self.adata, svd_solver='arpack')

        print('computing neighbors..')
        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, metric='cosine', n_pcs=n_pcs)

        print('performing umap...')
        sc.tl.umap(self.adata, min_dist=min_dist)  # , init_pos=adata.obsm['X_pca'])

        print('performing clustering...')
        sc.tl.leiden(self.adata, resolution=resolution)


        if labels != None:
            print('supervised umap...')

            embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist).fit_transform(self.adata.X, y=self.metadata[labels+'_int'])
            self.adata.obsm['X_umap_supervised'] = embedding

        self.feat_rank = {}

        for cat in categories:
            sc.tl.rank_genes_groups(self.adata, cat, method='wilcoxon')

            self.feat_rank[cat] = {'feature names':self.adata.uns['rank_genes_groups']['names'],
            'scores':self.adata.uns['rank_genes_groups']['scores'],
            'logfoldchanges':self.adata.uns['rank_genes_groups']['logfoldchanges'],
            'pvals_adj':self.adata.uns['rank_genes_groups']['pvals_adj']}



    def show_cellEmbed(self, label, embed_method, size):

        labels = self.adata.obs[label].values
        labels_unique = list(set(labels))
        labels_num = []

        for l in labels:
            labels_num.append(labels_unique.index(l))
        embedding = self.adata.obsm[embed_method]

        plt.close()
        fig, ax = plt.subplots(1, figsize=(5, 4))

        plt.scatter(embedding[:,0],embedding[:,1], s=size, c=labels_num, cmap='rainbow', alpha=1)
        plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(len(labels_unique)+1)-0.5)
        cbar.set_ticks(np.arange(len(labels_unique)))
        cbar.set_ticklabels(labels_unique)

        plt.show()


    def show_cellEmbed_label(self, label, classes, embed_method, size):
        
        data = self.metadata[label]
        labels_show = np.array(['others'] * data.shape[0])
        labels_show[data==classes] = classes
        labels_num = []
        for l in labels_show:
            if l == 'others':
                labels_num.append(0)
            else:
                labels_num.append(1)
        embedding = self.adata.obsm[embed_method]

        plt.close()
        fig, ax = plt.subplots(1, figsize=(5, 4))
        
        plt.scatter(embedding[:,0],embedding[:,1], s=size, c=labels_num, cmap='Reds', alpha=0.7, vmax=1.3)
        plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(3)-0.5)
        cbar.set_ticks(np.arange(2))
        cbar.set_ticklabels(['others',classes])
        plt.show()



    def show_featrank(self, label, region, rank_by, num_select, num_show):

        df= pd.DataFrame({'feature names':self.feat_rank[label]['feature names'][region],
                      'scores':self.feat_rank[label]['scores'][region],
                        'logfoldchanges':self.feat_rank[label]['logfoldchanges'][region],
                         'pvals_adj':-np.log10(self.feat_rank[label]['pvals_adj'][region])})

        df_select = df.sort_values(by=rank_by, ascending=False).iloc[:num_select+20]
        
        display(df_select.iloc[num_show:(20+num_show)])

        return df_select



    def show_featdist(self, features, embed_method, size):

        embedding = self.adata.obsm[embed_method]

        plt.close()
        fig, ax = plt.subplots(1, figsize=(5, 4))

        plt.scatter(embedding[:,0],embedding[:,1], s=size, c=self.intens_mtx[features], cmap='Reds', alpha=1)
        plt.setp(ax, xticks=[], yticks=[])
        plt.colorbar()

        plt.show()



    def LipidMaps_annotate(mass_list,adducts,tolerance,site_url):
        
        Data = []
        matched = []
        unmatched = []
        
        for mass in mass_list:
            Data_ = []
            for adduct in adducts:
                url = site_url+'/{}/{}/{}'.format(mass,adduct,tolerance)
                
                urlData = requests.get(url).content.decode('utf-8')[7:-9]            
                rawData = pd.read_csv(io.StringIO(urlData),sep='\t',error_bad_lines=False,index_col=False)
                
                Data_.append(rawData)
                Data.append(rawData)
            df = pd.concat(Data_, ignore_index=True)
            
            if df.empty:
                unmatched.append(mass)
            else:
                matched.append(mass) 
                
        annot_df = pd.concat(Data, ignore_index=True)
        return annot_df, matched, unmatched

