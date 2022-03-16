# SVM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.metrics import classification_report, confusion_matrix  
import joblib
from sklearn.metrics import roc_curve, auc
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import subplots,scatter
import seaborn as sns
matplotlib.interactive(True)


def rank_sum_test(x, y, features, N):
    dim = x.shape[1]
    sample_num = min(x.shape[0], y.shape[0])
    S = []
    P = []
    for i in range(dim):
        s, p = ranksums(x[:, i], y[:, i])
        S.append(s)
        P.append(p)

    S = np.asarray(S)
    P = np.asarray(P)

    P_order = P.argsort()[-N:]
    # P_ranks = P_order.argsort()
    feature_order = features[P_order]

    return P_order, feature_order, P


def SVM_classi(X_train, y_train, kernel_, param_grid, if_probability, cores):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #     data_dict = {'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test}

    grid_search = GridSearchCV(svm.SVC(kernel=kernel_), param_grid, cv=5, n_jobs=cores)

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    svclassifier = SVC(kernel=kernel_, C=best_params['C'], gamma=best_params['gamma'],probability=if_probability)
    svclassifier.fit(X_train, y_train)

    return svclassifier

def LR_classi(X_train, y_train, param_grid, cores):
    
    grid_search = GridSearchCV(linear_model.LogisticRegression(), param_grid, cv=10, n_jobs=cores)

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    lrclassifier = linear_model.LogisticRegression(C=best_params['C'], penalty=best_params['penalty'])
    lrclassifier.fit(X_train, y_train)

    return lrclassifier

def predict(X_test, y_test, classifier):
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    # sns.heatmap(cm,annot=True,fmt="d")
    return y_pred, cm, report_dict


def roc(classifier, X_test, y_test, n_classes):
    y_score = classifier.decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr['0'], tpr['0'], _ = roc_curve(y_test, y_score)
    roc_auc['0'] = auc(fpr['0'], tpr['0'])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return fpr, tpr, roc_auc


def test_features_num(group1, group2, data_dict, features_list, feature_num, param_grid, kernel,classifier_type):
    
    report_list = []
    y_pred_list = []
    cm_list = []
    fpr_list = []
    tpr_list = []
    roc_auc_list = []

    X_train, X_test, y_train, y_test = data_dict['X_train'], data_dict['X_test'], data_dict['y_train'], data_dict[
        'y_test']
    O, F, P = rank_sum_test(group1, group2, features_list, len(features_list))
    
    for i in feature_num:
        
        print('now testing '+str(i)+' features')
        #print(F)
        X_train_RS, X_test_RS = X_train[:, O[:i]], X_test[:, O[:i]]
        
        if classifier_type == 'SVM':
            classifier = SVM_classi(X_train_RS, y_train, kernel_=kernel, param_grid=param_grid, cores=8,if_probability=True)
        if classifier_type == 'LR':
            classifier = LR_classi(X_train_RS, y_train, param_grid=param_grid, cores=8)
        else:
            print('No argument for classifier found')
            
        y_pred, cm, report_dict = predict(X_test_RS, y_test, classifier)

        report_list.append(report_dict)
        y_pred_list.append(y_pred)
        cm_list.append(cm)

        fpr, tpr, roc_auc = roc(classifier, X_test_RS, y_test, 2)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)

        print('Finished testing with', i, 'features')
        print(report_dict['0.0']['precision'])
        print(report_dict['1.0']['precision'])

    return report_list, y_pred_list, cm_list, fpr_list, tpr_list, roc_auc_list

def test_features_num_rf(data_dict, ranked_features_index, feature_num, param_grid, kernel,cores):
    
    report_list = []
    y_pred_list = []
    cm_list = []
    fpr_list = []
    tpr_list = []
    roc_auc_list = []

    X_train, X_test, y_train, y_test = data_dict['X_train'], data_dict['X_test'], data_dict['y_train'], data_dict[
        'y_test']
    
    for i in feature_num:
        feature_index = ranked_features_index[:i]
        print('now testing '+str(i)+' features')
        X_train_RS, X_test_RS = X_train[:, feature_index], X_test[:, feature_index]

        classifier = SVM_classi(X_train_RS, y_train, kernel_=kernel, param_grid=param_grid, cores=cores,if_probability=True)
            
        y_pred, cm, report_dict = predict(X_test_RS, y_test, classifier)

        report_list.append(report_dict)
        y_pred_list.append(y_pred)
        cm_list.append(cm)

        fpr, tpr, roc_auc = roc(classifier, X_test_RS, y_test, 2)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)

        print('Finished testing with', i, 'features')
        print(report_dict['0.0']['precision'])
        print(report_dict['1.0']['precision'])

    return report_list, y_pred_list, cm_list, fpr_list, tpr_list, roc_auc_list

def test_features_sensitivity(data_dict, feature_list, feature_index, param_grid, kernel, cores):
    
    features = []
    prob = []
    
    X_train, X_test, y_train, y_test = data_dict['X_train'], data_dict['X_test'], data_dict['y_train'], data_dict['y_test']    
    X_train_RS, X_test_RS = X_train[:, feature_index], X_test[:, feature_index]
    classifier = SVM_classi(X_train_RS, y_train, kernel_=kernel, param_grid=param_grid, if_probability= True, cores=cores)
    y_pred_prob = classifier.predict_proba(X_test_RS)[y_test==1]
    
    for i in range(len(feature_index)):
        
        features.append(feature_list[feature_index[i]])
        index_selected = np.delete(feature_index,i)
        
        print('perturbing feature {} m/z'.format(feature_list[feature_index[i]]))
        
        X_train_selected, X_test_selected = X_train[:, index_selected], X_test[:, index_selected]
        classifier_selected = SVM_classi(X_train_selected, y_train, kernel_=kernel, param_grid=param_grid, if_probability= True, cores=cores)
        
        y_pred_prob_selected = classifier_selected.predict_proba(X_test_selected)
        
        prob.append(y_pred_prob_selected[y_test==1].mean(axis=0)[1]-y_pred_prob.mean(axis=0)[1])
        
    features = np.asarray(features)
    prob = np.asarray(prob)
    
    return features, prob

def plot_coefficients(classifier, feature_names, top_features=40):
    
    coef = classifier.coef_.ravel()
    index_sorted = np.argsort(coef)
    coef_sorted = coef[index_sorted]
    feature_names = np.array(feature_names)
    feature_names_sorted = feature_names[index_sorted]
    top_positive_coefficients = abs(coef_sorted[coef_sorted>0])[-top_features:]
    positive_features = feature_names_sorted[coef_sorted>0][-top_features:]
    top_negative_coefficients = abs(coef_sorted[coef_sorted<0])[::-1][-top_features:]
    negative_features = feature_names_sorted[coef_sorted<0][::-1][-top_features:]
    #top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    fig,axes=subplots(1,2,figsize=(12, 14))
    fig.suptitle('Important features for astrocytes vs neurons based on classification',fontsize=16)
    fig.text(0.5, 0.04, 'arbitrary SVM coefficients', ha='center',fontsize=12)
    fig.text(0.04, 0.5, 'm/z values', va='center', rotation='vertical',fontsize=12)
    ax=axes.ravel()
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    #colors = ['red' if c < 0 else 'blue' for c in coef[top_positive_coefficients]]
    ax[0].barh(np.arange(top_negative_coefficients.shape[0]), top_negative_coefficients, color='r')
    ax[0].set_yticks(np.arange(0, top_negative_coefficients.shape[0]))
    ax[0].set_yticklabels(negative_features)
    ax[0].set_xticklabels([])
    ax[1].barh(np.arange(top_positive_coefficients.shape[0]), top_positive_coefficients, color='g')
    ax[1].set_yticks(np.arange(0, top_positive_coefficients.shape[0]))
    ax[1].set_yticklabels(positive_features)
    ax[1].set_xticklabels([])
    #fig.legend(['astrocytes','neurons'])
    
    fig2,axes2 = plt.subplots(2,1,figsize=(20,10))
    ax = axes2.ravel()
    arkerline, stemlines, baseline = ax[0].stem(negative_features, top_negative_coefficients, '-.')
    ax[0].set_yticklabels([])
    ax[0].set_yticks([])
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].set_ylabel('feature importance',fontsize=14)
    ax[0].set_xlim([500,1000])
    baseline.set_xdata([500,1000])
    #ax[0].set_title('feature relative importance for astrocytes',fontsize=16)
    arkerline, stemlines, baseline = ax[1].stem(positive_features, top_positive_coefficients, '-.')
    ax[1].set_yticklabels([])
    ax[1].set_yticks([])
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].set_ylabel('feature importance',fontsize=14)
    ax[1].set_xlabel('m/z',fontsize=14)
    ax[1].set_xlim([500,1000])
    baseline.set_xdata([500,1000])
    #ax[1].set_title('feature relative importance for neurons',fontsize=16)


