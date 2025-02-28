import os
import random
import time
import warnings
from collections import defaultdict

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from river import ensemble, feature_selection
from river import stream, linear_model
from river.naive_bayes import GaussianNB
from river.tree import HoeffdingAdaptiveTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def print_metrics(model_name, data, y, y_pred, time_in, labels=None):
    if labels is None:
        labels = []

    accuracy_value = accuracy_score(y, y_pred)
    precision_score_values = precision_score(y, y_pred, average=None, labels=labels)
    recall = recall_score(y, y_pred, average=None, labels=labels)
    f1_score_values = f1_score(y, y_pred, average=None, labels=labels)
    auc = roc_auc_score(y, y_pred)

    latex_result = "\\textsc{" + model_name + "}" + " & {:.2f}".format(accuracy_value * 100) \
                   + " & {:.2f}".format(auc * 100) \
                   + " & {:.2f}".format(precision_score_values.mean() * 100) \
                   + " & {:.2f}".format(precision_score_values[0] * 100) \
                   + " & {:.2f}".format(precision_score_values[1] * 100) \
                   + " & {:.2f}".format(recall.mean() * 100) \
                   + " & {:.2f}".format(recall[0] * 100) \
                   + " & {:.2f}".format(recall[1] * 100) \
                   + " & {:.2f}".format(f1_score_values.mean() * 100) \
                   + " & {:.2f}".format(f1_score_values[0] * 100) \
                   + " & {:.2f}".format(f1_score_values[1] * 100) \
                   + " & {:.2f}".format(time_in) \
                   + "\\\\"

    return {"model_hyper_params": data["model_hyper_params"], "latex": latex_result}


def instance_distance(x_original: dict, x_modified: dict):
    dist = 0.0
    dist_dic = {}
    for feature in x_original:
        dist_aux = abs(x_modified[feature] - x_original[feature])
        if x_modified[feature] - x_original[feature] > 0:
            dist_dic[feature] = x_modified[feature] - x_original[feature]
        dist += dist_aux
    return dist, dist_dic


def find_counterfactual(
        model,
        x_original: dict,
        desired_label,
        n_iter: int = 1000):
    proba_return = 0
    best_cf = None
    best_distance = float('inf')
    best_dist_dic = {}

    for _ in range(n_iter):
        x_candidate = x_original.copy()

        feature_groups = defaultdict(list)
        for feature in x_candidate.keys():
            prefix = feature.split('_')[0]
            feature_groups[prefix].append(feature)

        for prefix, group in feature_groups.items():
            if desired_label == 1:
                valid_options = [f for f in group if not f.endswith('_No')]
            else:
                valid_options = [f for f in group if not f.endswith('_Yes')]

            chosen_feature = random.choice(valid_options)

            for feature in group:
                x_candidate[feature] = 1 if feature == chosen_feature else 0

        pred = model.predict_one(x_candidate)
        proba_desired = model.predict_proba_one(x_candidate)[desired_label]

        if pred == desired_label and proba_desired > 0.5:
            dist, dist_dic = instance_distance(x_original, x_candidate)
            if dist < best_distance:
                proba_return = proba_desired
                best_dist_dic = dist_dic
                best_distance = dist
                best_cf = x_candidate.copy()

    return proba_return, best_cf, best_distance, best_dist_dic


def balanced_samples(dataset, n_samples):
    under_sampler = RandomUnderSampler(random_state=1)
    dataset, _ = under_sampler.fit_resample(dataset, dataset["target"])
    dataset = dataset.reset_index(drop=True)
    dataset = dataset.sort_values(by=['timestamp'], ascending=True)
    dataset = dataset.reset_index(drop=True)
    print(dataset["target"][0:n_samples].value_counts())
    return dataset


def start_river_analysis(data):
    path = data["path"]
    model_name = data["model_name"]
    model_hyper_params = data["model_hyper_params"]
    hyper_params_evaluating = data["hyper_params_evaluating"]
    columns_to_drop = data["columns_to_drop"]
    target = data["target"]
    labels = data["labels"]
    window_step = data["window_step"]
    variance = data["variance"]

    selector_bool = data["selector"]
    explainability = data["explainability"]
    verbose = data["verbose"]
    clustering_boolean = False
    if verbose:
        print("Start river analysis")
    model_to_analyse = None
    if model_name == "arfc":
        model_to_analyse = ensemble.AdaptiveRandomForestClassifier(n_models=model_hyper_params["n_models"],
                                                                   max_features=model_hyper_params["max_features"],
                                                                   lambda_value=model_hyper_params["lambda_value"],
                                                                   seed=1)
    elif model_name == "hatc":
        model_to_analyse = HoeffdingAdaptiveTreeClassifier(max_depth=model_hyper_params["max_depth"],
                                                           tie_threshold=model_hyper_params["tie_threshold"],
                                                           max_size=model_hyper_params["max_size"],
                                                           seed=1)
    if model_name == "log":
        model_to_analyse = linear_model.LogisticRegression(l1=model_hyper_params["l2"],
                                                           intercept_lr=model_hyper_params["intercept_lr"], )
    if model_name == "alma":
        model_to_analyse = linear_model.ALMAClassifier(alpha=model_hyper_params["alpha"],
                                                       B=model_hyper_params["B"],
                                                       C=model_hyper_params["C"])
    elif model_name == "gnb":
        model_to_analyse = GaussianNB()

    selector = feature_selection.VarianceThreshold(threshold=variance)

    classifier_model = {"model": model_to_analyse,
                        "elements": 0}

    dataset_sensores = pd.read_csv(path, engine="pyarrow")

    if verbose:
        print(dataset_sensores.shape)

    dataset_sensores = dataset_sensores.dropna()

    if hyper_params_evaluating:
        dataset_sensores = dataset_sensores[0:int(len(dataset_sensores) * 0.10)]
        print(dataset_sensores.shape)

    if verbose:
        print(dataset_sensores.shape)

    dataset_sensores = dataset_sensores.iloc[:-1:window_step]

    y = dataset_sensores[target]
    X = dataset_sensores[dataset_sensores.columns.difference(columns_to_drop)]

    list_y_pred = []
    list_y = []
    init_value = time.time()
    count_train = 0
    list_x_river = []
    list_x_river_window = []

    for x_rive_new, y_river in stream.iter_pandas(X, y):
        if selector_bool:
            x_river = selector.learn_one(x_rive_new).transform_one(x_rive_new)
        else:
            x_river = x_rive_new

        if clustering_boolean:
            list_x_river.append(x_river)
            list_x_river_window.append(x_river)

        if classifier_model["elements"] == 0 and clustering_boolean:
            classifier_model["model"] = classifier_model["model"].learn_one(x_river)

        y_pred = classifier_model["model"].predict_one(x_river)
        y_pred_proba = 0
        if y_pred is not None:
            y_pred_proba = classifier_model["model"].predict_proba_one(x_river)[y_pred]

        if verbose:
            print(str(y_pred) + " - " + str(y_river))
        if verbose:
            print(classifier_model["elements"])

        if y_pred is not None:
            list_y_pred.append(y_pred)
            list_y.append(y_river)

        try:
            count_train = count_train + 1
            if not clustering_boolean:
                classifier_model["model"] = classifier_model["model"].learn_one(x_river, y_river)
            else:
                classifier_model["model"] = classifier_model["model"].learn_one(x_river)

            classifier_model["elements"] = classifier_model["elements"] + 1
        except Exception as e:
            if verbose:
                print(e)
            selector = feature_selection.VarianceThreshold(threshold=variance)

        # Counterfactual
        if explainability:
            if 0.8 < y_pred_proba < 1.0 and y_river == y_pred:
                if y_river == 1:
                    desired_label = 0
                else:
                    desired_label = 1

                proba_return, best_cf, best_distance, best_dist_dic = find_counterfactual(classifier_model["model"],
                                                                                          x_rive_new, desired_label,
                                                                                          1000)
                print(proba_return, best_cf, best_distance, best_dist_dic)

    if verbose:
        print("Train count: " + str(count_train))

    return print_metrics(model_name, data, list_y, list_y_pred, time.time() - init_value, labels=labels)


def run_paper_experiments(scenario, list_models):
    target = "target"
    labels = [0, 1]
    only_avg = False
    columns_to_drop = ['target', 'timestamp']

    window_step = 1

    variance = 0.079

    path = ""
    if scenario == 0:
        path = "datasets/dataset_processed.csv"

    model_hyper_params = {
        "n_models": 100, "max_features": "sqrt", "lambda_value": 100,
        "max_depth": None, "tie_threshold": 0.05, "max_size": 50,
        "alpha": 0.5, "B": 0.6, "C": 1.4,
        "l2": 0.0, "intercept_lr": 0.01,
    }

    for z in list_models:
        data = {"path": path, "model_name": z, "model_hyper_params": model_hyper_params,
                "hyper_params_evaluating": False,
                "columns_to_drop": columns_to_drop,
                "only_avg": only_avg,
                "target": target,
                "labels": labels,
                "window_step": window_step,
                "variance": variance,
                "selector": True, "explainability": False, "verbose": False}

        complete_info_result = start_river_analysis(data)
        print(complete_info_result["latex"])


if __name__ == '__main__':
    list_models = ["gnb", "log", "alma", "hatc", "arfc"]
    run_paper_experiments(scenario=0, list_models=list_models)
