import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv(r"D:\Data Science\GitHub\Model\Airlines_Flights_Data.csv")

# Drop unwanted columns safely
cols_to_drop = ['index', 'flight', 'arrival_time', 'duration', 'days_left']
df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)

# Label encoding for categorical columns
label = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = label.fit_transform(df[col])

# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Save model file (.pkl)
joblib.dump(model, "model.pkl", compress=3)
print("Model saved as model.pkl")
