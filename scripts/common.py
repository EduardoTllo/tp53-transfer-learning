"""Code shared by notebooks 04 to 09: load the data, normalize it, build the network and compare SDs (Pitman-Morgan).

Some names are in Spanish (cargar = load, fuente = source, armar_red = build network,
pesos_clase = class weights). They are kept as they are because the saved notebooks use them.
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FILES = {"TCGA-BRCA": "tcga_brca", "METABRIC": "metabric", "CPTAC": "cptac", "SMC 2018": "smc_2018"}


def cargar(cohorte):
    """Load one cohort: log2 expression (patients × genes) and labels (1 = TP53 mutated), in the same order."""
    x = pd.read_parquet(DATA / f"X_{FILES[cohorte]}.parquet")
    y = pd.read_csv(DATA / f"y_{FILES[cohorte]}.csv", index_col=0).loc[x.index, "y"]
    return x, y


def zscore(x, ref):
    """z-score of each gene, with the mean and SD of the rows in `ref`.

    Missing values are first filled with the median of `ref`. If a gene does not vary in `ref`, its SD is set to 1.
    Source cohorts use ref = all their patients. Target cohorts use ref = the patients outside the exam (no labels).
    """
    x = x.fillna(x.loc[ref].median())
    mean, sd = x.loc[ref].mean(), x.loc[ref].std().replace(0, 1)
    return ((x - mean) / sd).fillna(0).astype(np.float32)


def fuente():
    """Source data: TCGA-BRCA and METABRIC, each normalized with all its own patients, then joined.

    Returns X (DataFrame), y (Series) and the cohort of each row (Series).
    """
    parts = []
    for name in ("TCGA-BRCA", "METABRIC"):
        x, y = cargar(name)
        parts.append((zscore(x, x.index), y, pd.Series(name, index=x.index)))
    return tuple(pd.concat(p) for p in zip(*parts))


# False warning from the macOS math library (it appears even with random numbers). Results are valid.
warnings.filterwarnings("ignore", message=".*encountered in matmul")

# Keras runs on PyTorch. TensorFlow 2.20 and 2.21 crash on import on macOS 26.4.
os.environ.setdefault("KERAS_BACKEND", "torch")


def armar_red(capas, dropout, tasa, optimizador):
    """Dense network (MLP): 18,040 genes → hidden layers with ReLU and dropout → probability of TP53 mutated.

    `capas` (layers) is a list with the neurons of each hidden layer, for example [512, 128].
    `optimizador` (optimizer) is "adam" or "sgd" (SGD uses momentum 0.9).
    """
    import keras

    layers = [keras.Input((18040,))]
    for n in capas:
        layers += [keras.layers.Dense(n, activation="relu"), keras.layers.Dropout(dropout)]
    layers.append(keras.layers.Dense(1, activation="sigmoid"))
    opt = keras.optimizers.Adam(tasa) if optimizador == "adam" else keras.optimizers.SGD(tasa, momentum=0.9)
    model = keras.Sequential(layers)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")])
    return model


def pesos_clase(y):
    """Weight of each class in the loss. A mistake on the less frequent class costs more."""
    y = np.asarray(y)
    return {0: len(y) / (2 * (y == 0).sum()), 1: len(y) / (2 * (y == 1).sum())}


def pitman_morgan(a, b):
    """Pitman-Morgan test: compares the SD of two paired lists of AUCs (same repetitions).

    Returns the two-sided p value and a signed z: positive when `a` has the smaller SD.
    """
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    m = len(a)
    r = np.corrcoef(a + b, a - b)[0, 1]  # r > 0 when the variance of a is larger than the variance of b
    # shortcut: if one strategy has SD = 0 (same AUC in every repetition), r = ±1 and z would be infinite. It is capped at |z| ≈ 71 (m = 100).
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt((m - 2) / (1 - r**2))
    p = 2 * stats.t.sf(abs(t), m - 2)
    # z with the Fisher transformation. It matches the t test, but it stays finite when p < 1e-308.
    return p, -np.arctanh(r) * np.sqrt(m - 3)

