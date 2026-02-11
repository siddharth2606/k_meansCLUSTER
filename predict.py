import joblib

model = joblib.load("k_meansCLUSTER/src/KMEANS_model.pkl")
prediction = model.predict([[4.7,3.2,1.3,0.2]])
print(prediction)