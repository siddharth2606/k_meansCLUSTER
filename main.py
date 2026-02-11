from data_loader import load_data

dataset = load_data("k_meansCLUSTER/data/iris_with_names.csv")

new_data = dataset.drop(columns="species",axis=1)
print(new_data.head())