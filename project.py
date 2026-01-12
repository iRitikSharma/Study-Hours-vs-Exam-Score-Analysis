import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# load dataset
data = pd.read_csv("Machine Learning/project/student_performance_dataset.csv")

# input & output
X = data[["hours_studied_per_week"]]
y = data[["marks"]]

model = LinearRegression()
model.fit(X,y)

predicted_scores = model.predict(X)

# Valid Regression metrics
mae = mean_absolute_error(y,predicted_scores)
mse  = mean_squared_error(y,predicted_scores)
rmse = np.sqrt(mse)
r2 = r2_score(y,predicted_scores)


# show results
print(f"""
Model Evaluation Metrics
------------------------
MAE  : {mae:.2f}
MSE  : {round(mse,2)}
RMSE : {round(rmse,2)}
R^2 Score (Model Accuracy): {round(r2,4)}  

""") #r2 better when closer to 1

plt.figure(figsize=(6,4))   #width, height
plt.hist(y,bins=5, color="yellow" , edgecolor ="black")
plt.title("Distribution of FINAL EXAM SCORES")
plt.xlabel("Final Exam Score")
plt.ylabel("No. of Students")
plt.grid(True,color = "grey")
plt.show()



# scatter plot + regression line

plt.figure(figsize=(6,4))   #width, height
plt.scatter(X ,y, color="#ba4242", edgecolors="black", label = "actual scores")
plt.plot(X,predicted_scores, color ="green" , label="predicted Scores (Regression Line)")
plt.title("Model Prediction VS Actual Scores")
plt.xlabel("Study Hours per Week")
plt.ylabel("Final Output")
plt.grid(True,color = "grey")
plt.show()



new_hours = 9
predicted_new_score = model.predict([[new_hours]])
print(f"predicted final score for {new_hours} is {predicted_new_score}")
