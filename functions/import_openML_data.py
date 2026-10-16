# import data
# the mian contents of this script are taken from https://github.com/eepperly/Robust-randomized-preconditioning-for-kernel-ridge-regression/blob/main/utils/download_data.py by Ethan Epperly

import openml
import pandas
import os
import numpy as np
import random
import math

datasets_openml = {
    "ACSIncome": {
        "id": 43141,
        "target": "PINCP",
        "test_proportion": 0.2,
    },
    "Airlines_DepDelay_1M": {
        "id": 42721,
        "target": "DepDelay",
        "test_proportion": 0.2,
    },
    "Click_prediction_small": {
        "id": 1218,
        "target": "click",
        "test_proportion": 0.2,
    },
    "COMET_MC_SAMPLE": {
        "id": 23395,
        "target": "label",
        "test_proportion": 0.2,
    },
    "creditcard": {
        "id": 1597,
        "target" : "Class",
        "test_proportion": 0.2,
    },
    "diamonds": {
        "id": 42225,
        "target": "price",
        "test_proportion": 0.2,
    },
    "HIGGS" : {
        "id": 42769,
        "target": "target",
        "ntr": 500000,
        "nts": 500000,
    },
    "hls4ml_lhc_jets_hlf": {
        "id": 42468,
        "target": "class",
        "test_proportion": 0.2
    },
    "jannis": {
        "id": 44131,
        "target": "class",
        "test_proportion": 0.2,
    },
    "Medical-Appointment": {
        "id": 43617,
        "target": "show",
        "test_proportion": 0.2,
    },
    "MiniBoonNE": {
        "id": 41150,
        "target": "signal",
        "test_proportion": 0.2,
    },
    "MNIST": {
        "id": 554,
        "target": "class",
        "ntr": 60000,
        "nts": 10000,
    },
    "santander": {
        "id": 42435,
        "target": "target",
        "test_proportion": 0.2,
        "remove_features": ["ID_code"],
    },
    "volkert": {
       "id": 41166,
        "target": "class",
        "test_proportion": 0.2,
    },
    "yolanda": {
        "id": 42705,
        "target": "101",
        "test_proportion": 0.2,
    },
}

def download_data(name, path="data"):
    parent = os.path.dirname(os.path.abspath(os.getcwd()))
    filepath = os.path.join(parent, path, name)
    if os.path.isfile(filepath):
        print(f"{name} data already exists...")
        return np.load(filepath)

    print(f"File not found. Downloading {name}...")
    dataset = openml.datasets.get_dataset(datasets_openml[name]['id'])
    X, _, _, attr = dataset.get_data(dataset_format="dataframe")
    Xmat = X.to_numpy()
    np.save(filepath, Xmat)
    return Xmat

def download_and_mat(name, m, path="data", kername="RBF", theta=[1.0, 1.0]):
    Xmat = download_data(name, path)
    M = Xmat.shape[0]
    samples = random.sample(range(M), m)
    Xsamp = Xmat[samples, :]

    match kername:
        case "RBF":
            def ker(x, y, theta):
                    # theta[0] = sigma_f
                    # theta[1] = l
                t = np.dot(x-y, x-y)
                out = theta[0]**2 * math.exp(-1/(2*theta[1]) * t)
                return out

            def ker_p(x, y, theta):
                    # theta[0] = sigma_f
                    # theta[1] = l
                out = np.zeros(2)
                t = -1/(2*theta[1]) * np.dot(x-y,x-y)
                out[0] = 2*theta[0] * math.exp(t)
                out[1] = np.dot(x-y, x-y) * 1 / (2*theta[1]**2) * theta[0]**2 * math.exp(t)
                return out

    K = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            if i>j:
                continue
            if i==j:
                K[i,j] = ker(Xsamp[i,:], Xsamp[j,:], theta)
            else:
                K[i,j] = ker(Xsamp[i,:], Xsamp[j,:], theta)
                K[j,i] = ker(Xsamp[i,:], Xsamp[j,:], theta)

    return K
