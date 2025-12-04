# ✈️ Flight Price Prediction – Machine Learning Project  
Predict Flight Ticket Prices Using Decision Tree Regression & Streamlit

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![ML](https://img.shields.io/badge/Machine%20Learning-Decision%20Tree-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)

---

## 🚀 Live Demo  
🔗 **Streamlit App:** 
https://87bpufzejguifvkommxgph.streamlit.app/
---

## 📌 Project Overview  

This project aims to **predict flight ticket prices** based on various airline and travel-related features.  
It includes:

- Data cleaning & preprocessing  
- Decision Tree Regression model training  
- Saving the trained model as `model.pkl`  
- Streamlit web app for predictions  
- Deployment on cloud  
- Full GitHub-hosted repository  

---


## 📊 Dataset Information  

The dataset contains the following columns:

| Column Name        | Description |
|--------------------|-------------|
| index              | Index column (not used for model training) |
| airline            | Airline name |
| flight             | Flight number |
| source_city        | Departure city |
| departure_time     | Time of day (Morning/Evening) |
| stops              | Number of stops (zero/one/two+) |
| arrival_time       | Arrival time category |
| destination_city   | Destination city |
| class              | Travel class (Economy/Business) |
| duration           | Flight duration in hours |
| days_left          | Days left before departure |
| price              | Final ticket price (target variable) |

---

### 🗑️ Columns Removed Before Training  

The following columns were dropped because they do not contribute to model performance or may leak information:

- `index`
- `flight`
- `arrival_time`
- `duration`
- `days_left`

---

## 🤖 Model Training – Overview  

The model is built using a **Decision Tree Regressor**.

Training workflow includes:

- Preprocessing the dataset  
- Encoding categorical features using LabelEncoder  
- Splitting into training & testing sets  
- Training Decision Tree Regression  
- Evaluating model using **MSE** and **R² Score**  
- Saving trained model as `model.pkl`  
- Using the model in a Streamlit prediction UI  

---

## 🖥 Streamlit App  

The Streamlit application allows users to enter:

- Airline  
- Source & Destination  
- Departure time  
- Number of stops  
- Class (Economy/Business)  

and instantly predicts the **ticket price** using the trained model.

---

## 🛠️ Technologies Used  

- Python  
- Pandas, NumPy  
- Scikit-Learn  
- Decision Tree Regression  
- Streamlit  
- Pickle  
- GitHub  

---

## 🙋‍♂️ Author  

Update with your details:

- **Name:** Urvi Patel  
- **GitHub:** urvi45    

---

## ⭐ Support  

If you like this project, support it by giving the repository a **star** ⭐ on GitHub!

