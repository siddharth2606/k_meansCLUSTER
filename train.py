from data_loader import load_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans

dataset = load_data("k_meansCLUSTER/data/iris_with_names.csv")
x = dataset.iloc[:,:-1]

kmc = KMeans(n_clusters=3,random_state=42)
kmc.fit(x)

joblib.dump(kmc,"KMeans_model.pkl")