import numpy as np
import pandas as pd
import anndata
import seaborn as sns
import scanpy as sc
from numpy.linalg import svd
from sklearn.cluster import KMeans

from dcv import *

sc.settings.set_figure_params(dpi=80, facecolor='white')

#******************1.Prepare data******************
intens_df = loadmatfile('Dan_data/vesicles_data.mat')
mass_list = loadmat('Dan_data/acc_masses.mat')['acc_masses'][0]
names = loadmat('Dan_data/names.mat')['names'][0]

#apply filters to select features with cell number

data_filtered = intens_df.iloc[:,intens_df.astype(bool).sum(axis=0).values>30]
data_filtered = data_filtered.iloc[:,data_filtered.columns>1100]

print('filtered intensity matrix with shape {}'.format(data_filtered.shape))

data_filtered.head(10)

#normalize feature values

norm_factors = np.sqrt(np.mean(data_filtered.replace(0,np.NaN)**2,axis=1)).values
norm_factors = norm_factors.reshape(data_filtered.shape[0],1)
data_filtered = np.divide(data_filtered,norm_factors)
features = data_filtered.columns.values


#******************2.Perform SVD analysis******************
U,D,V = svd(data_filtered, full_matrices=False)

#cumulative singular values
S_cumu = []
for i in range(len(D)):
    S_cumu.append(np.sqrt(np.sum(D[:i]**2))/np.sqrt(np.sum(D**2)))

#reconstruction error with different choices of Ks
K = [30,50,100,150,200]
baseline = []
A = data_filtered.values
norm_X = np.linalg.norm(A, 'fro')
for k in K:
    proj = np.dot(U[:,:k], U[:,:k].T)
    Ak = np.dot(proj,A)
    baseline.append(np.linalg.norm(A - Ak, 'fro'))
    errors = baseline/norm_X

#******************3.CX decomposition******************
import random 

N = [100,150,200,250]
baseline_CX = []
baseline_CX_mean = []
baseline_CX_std = []

lev_150 = comp_lev(A,150,1)

for n in N:
    C, X, lev = cx_decomp(A, 100, n)
    p = lev/k
    baseline_CX.append(np.linalg.norm(A - np.dot(C,X), 'fro'))
    
    baseline_CX_sample = []
    for j in range(5):
        
        colInd = random.sample(list(np.linspace(0,A.shape[1]-1,A.shape[1]-1).astype(int)),n)

        C = np.dot(A[:,colInd], np.diag(1/p[colInd]))
        X = np.dot(np.linalg.pinv(C), A)
        baseline_CX_sample.append(np.linalg.norm(A - np.dot(C,X), 'fro'))
        
    baseline_CX_mean.append(np.mean(baseline_CX_sample))
    baseline_CX_std.append(np.std(baseline_CX_sample))
    
#     C, U, R = cur_decomp(A, 150, n,598)
#     baseline_CUR.append(np.linalg.norm(A - np.dot(np.dot(C,U),R), 'fro'))
    
errors_CX = baseline_CX/norm_X
baseline_CX_mean = baseline_CX_mean/norm_X
baseline_CX_std = baseline_CX_std/norm_X

print(errors_CX)


#******************4.Multivariate analysis******************
rank_idx = lev.argsort()[::-1]

#select top 200 features ranked by the leverage scores
data_selected = data_filtered.iloc[:,rank_idx[:200]]

features = data_selected.columns.values
X = (data_selected.loc[:, features].values)
adata = anndata.AnnData(X)

#perform multivariate analysis pipeline on the selected feature matrix
process(adata, n_pcs=15, min_dist=0.3, resolution = 0.6)

# extract pca coordinates
X_pca = adata.obsm['X_pca'] 

# kmeans with k=3
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_pca) 
adata.obs['kmeans'] = kmeans.labels_.astype(str)

#construct a pca dataframe with cluster assignments
pca_df = pd.DataFrame(adata.obsm['X_pca'][:,:2],columns=['PC1','PC2'])
pca_df['cluster'] = adata.obs['kmeans'].values.astype(int)

adata.var['mz'] = [a+' m/z' for a in features.astype(str)]
adata.var_names = [a+' m/z' for a in features.astype(str)]

#test the significant features in 3 individual clusters (wilcoxon)
sc.tl.rank_genes_groups(adata, 'kmeans', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=30, sharey=False,gene_symbols='mz',save='wilcoxon.pdf')

#features to visualize intensity distribution and violin plot 
mzs = ['1117.08 m/z','1197.18 m/z','1231.0 m/z','1243.69 m/z','1418.74 m/z','1671.81 m/z','1908.91 m/z','2304.28 m/z','2474.38 m/z','3435.88 m/z','4047.12 m/z']

sc.pl.pca(adata, color=mzs,s=60,save='feature_plot.pdf')
sc.pl.stacked_violin(adata, mzs, groupby='kmeans',figsize=(6,3),save='stackviolin.pdf')