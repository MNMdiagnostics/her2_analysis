import pandas as pd
import numpy as np

from tqdm.notebook import tqdm

from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score, cohen_kappa_score, f1_score, matthews_corrcoef, log_loss, plot_confusion_matrix, roc_auc_score, plot_roc_curve, balanced_accuracy_score, classification_report

def calculate_metrics(y_true, y_pred, y_prob, positive_class_label=1):
    y_bin = (y_true == positive_class_label)*1
    y_bin_pred = (y_pred == positive_class_label)*1

    precision = precision_score(y_bin, y_bin_pred)
    recall = recall_score(y_bin, y_bin_pred)
    accuracy = accuracy_score(y_bin, y_bin_pred)
    f1 = f1_score(y_bin, y_bin_pred)
    auc_roc = roc_auc_score(y_bin, y_prob)
    kappa = cohen_kappa_score(y_bin, y_bin_pred)
    mcc = matthews_corrcoef(y_bin, y_bin_pred)
    balanced_acc = balanced_accuracy_score(y_bin, y_bin_pred)
    result = pd.DataFrame({'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'AUC_ROC': auc_roc, 'Kappa': kappa, 'MCC': mcc, 'Balanced accuracy': balanced_acc}, index=[positive_class_label])        

    return result

def comparative_evaluation(datasets, estimators, cv, positive_class=1, threshold=0.5, random_state=None):

    result = pd.DataFrame({'Dataset': [], 'Classifier': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'AUC_ROC': [], 'Kappa': [], 'MCC': [], 'Balanced accuracy': []})

    pbar = tqdm(total=len(datasets)*cv.get_n_splits())

    for dataset_name, data_tuple in datasets.items():
        X, y = data_tuple

        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            for clf_string, classifier in estimators.items():
                run_result = run_single_test(classifier, X_train, y_train, X_test, y_test, positive_class, threshold)
                row = pd.Series([dataset_name, clf_string, *run_result], index=result.columns)
                result = result.append(row, ignore_index=True)

            pbar.update()

    pbar.close()

    result_melted = pd.melt(result, id_vars=['Dataset', 'Classifier'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC_ROC', 'Kappa', 'MCC', 'Balanced accuracy'], var_name='Metric', value_name='Value')

    return result_melted

def run_single_test(classifier, X_train, y_train, X_test, y_test, positive_class=None, threshold=0.5):
    model = classifier.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)
    pos_class_index = np.where(model.classes_ == positive_class)[0][0]
    neg_class_index = 1*(not pos_class_index)
    y_pred = np.array([get_class(prob, model.classes_[pos_class_index], model.classes_[neg_class_index], threshold) for prob in y_prob[:, pos_class_index]])

    single_run_result = classification_report(y_test, y_pred, output_dict=True)
    auc_roc = roc_auc_score((y_test == positive_class)*1, y_prob[:, pos_class_index])
    kappa = cohen_kappa_score((y_test == positive_class)*1, (y_pred == positive_class)*1)
    mcc = matthews_corrcoef((y_test == positive_class)*1, (y_pred == positive_class)*1)
    balanced_acc = balanced_accuracy_score((y_test == positive_class)*1, (y_pred == positive_class)*1)
    result_row = [single_run_result['accuracy'], single_run_result[str(positive_class)]['precision'], single_run_result[str(positive_class)]['recall'], single_run_result['macro avg']['f1-score'], auc_roc, kappa, mcc, balanced_acc]
    return result_row

def get_class(prob, pos_class_label=1, neg_class_label=0, threshold=0.5):
    if prob >= threshold:
        return pos_class_label
    else:
        return neg_class_label

def get_mean_results(results, by='Classifier'):
    return results.groupby(by=[by, 'Metric']).agg(['mean', 'std'])