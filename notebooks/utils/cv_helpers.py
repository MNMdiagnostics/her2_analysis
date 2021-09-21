import pandas as pd
import numpy as np

from plotnine import *

from sklearn.model_selection._split import RepeatedStratifiedKFold


def estimator_cv_evaluation(X, y, estimator_name, estimator_object, cv_scheme, scoring):
    """
    Perform cross-validation for given classifier with fixed parameters and collect the results of every run in the DataFrame
    
    :X (pd.DataFrame): Feature set
    :y (pd.Series): Targets for feature set
    :estimator_name (str): Name of estimator being evaluated in cross-validation
    :estimator_object: Object of estimator being evaluated (eg. RandomForestClassifier(), or SVC())
    :cv_scheme: scikit-learn.model_selection cross-validation object with parameters (eg. RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=0))
    :scoring (dict): Dictionary where key is a string of classifier name/ID and value is a sklearn.metrics object of wanted score
    """
    
    results = pd.DataFrame()
    run = 0
    for train_index, test_index in cv_scheme.split(X, y):
        run += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        fitted_model = estimator_object.fit(X_train, y_train)

        scoring_results = {score_name:score(estimator=fitted_model, X=X_test, y_true=y_test) for score_name, score in scoring.items()}
        one_run_result = pd.DataFrame(scoring_results, index=[0])
        one_run_result['Run'] = run
        one_run_result['Classifier'] = estimator_name
        
        results = results.append(one_run_result)
        
    return results


def plot_cv_estimators_comparison(comparison_results_df, metric, rotation=45, palette=("#1cbd9c", "#4FA775", "#a7d6bb"), title='', jitter=0.1, hjust=1, plot_type='violin'):
    """
    Plot results of estimator_cv_evaluation()
    
    :comparison_results_df (pd.DataFrame): Dataframe with CV results for each run per classifier
    :metric (str): Metric you want to plot
    """
    p = (ggplot(comparison_results_df, aes(x='Classifier', y=f'{metric}', fill='Classifier')) + \
        #  geom_violin() + \
         ggtitle(title) + \
         theme(axis_text_x=element_text(rotation=rotation, hjust=hjust)) + \
         scale_fill_manual(values=palette)
        )
    
    if plot_type == 'box':
        p = p + geom_boxplot()
    else:
        p = p + geom_violin()


    if jitter:
        p = p + geom_jitter(width=jitter)

    return p


def show_cv_results(cv_results, measures=['Accuracy', 'Precision', 'Recall', 'F1 Macro', 'F1 Weighted', 'Kappa']):
    """
    Shows dataframe with CV results from grid search
    
    :param cv_results: grid_search.cv_results_ object
    
    :return: pd.DataFrame with CV results
    """
    train_measures = ['mean_train_' + measure for measure in measures]
    test_measures = ['mean_test_' + measure for measure in measures]
    all_measures = train_measures + test_measures

    just_important_columns_df = pd.DataFrame(cv_results)[all_measures]
    cv_results_means_df = pd.DataFrame(just_important_columns_df.mean())
    cv_results_sd_df = pd.DataFrame(just_important_columns_df.std())
    
    if cv_results_sd_df.isna().all().item():
        cv_results_df = cv_results_means_df
        cv_results_df['SD'] = 0
    else:
        cv_results_df = pd.merge(cv_results_means_df, cv_results_sd_df, left_index=True, right_index=True)
    
    train_results = cv_results_df[cv_results_df.index.str.contains('train')]
    train_results.index = train_results.index.str.replace('mean_train_', '')
    test_results = cv_results_df[cv_results_df.index.str.contains('test')]
    test_results.index = test_results.index.str.replace('mean_test_', '')
    
    train_results = pd.concat([train_results], keys=['Train'])
    test_results = pd.concat([test_results], keys=['Test'])
    
    results_df = pd.concat([train_results, test_results])
    results_df.columns = ['Mean', 'SD']
    
    return results_df


class AugmentedRepeatedStratifiedKFold:
    def __init__(self, n_splits=2, n_repeats=10, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y, groups):
        X_add, y_add = X[groups], y[groups]
        X_base, y_base = X[~np.array(groups)], y[~np.array(groups)]

        base_cv = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)

        for train_i, test_i in base_cv.split(X_base, y_base):
            y_train_full = y_base.iloc[train_i].append(y_add)
            y_test_full = y_base.iloc[test_i]

            full_train_i = [i for i, idx in enumerate(y.index) if idx in y_train_full.index]
            full_test_i = [i for i, idx in enumerate(y.index) if idx in y_test_full.index]

            yield full_train_i, full_test_i

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats