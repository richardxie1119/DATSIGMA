import pandas as pd
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def loadmatfile(file_dir):
    
    f = loadmat(file_dir)
    names = f['data']['names'][0][0]
    name_list = [i[0] for i in names[0]]
    intens_matrix = f['data']['intens'][0][0].T
    mzs = f['data']['mzs'][0][0][0]

    print('Loaded intensity matrix with shape {}'.format(intens_matrix.shape))
    intens_df = pd.DataFrame(intens_matrix)
    intens_df = intens_df.set_index([pd.Index(name_list)])
    #intens_df[intens_df==0]=1
    #intens_df = np.log(intens_df)
    intens_df.columns = np.round(mzs,4)
    
    return intens_df


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.figure(figsize=(5,5))
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

