# Nuclear Medicine Dosimetry Modeller with AI-Based Individual Model Selection

**Author:** Annisa Rachman (2025)  
**LinkedIn:** [linkedin.com/in/annisarachman](https://linkedin.com/in/annisarachman)  
**GitHub:** [github.com/annrachman29](https://github.com/annrachman29)  

---

## Project Overview

This project is a **web application** designed to automate internal radiation dosimetry in nuclear medicine. It uses machine learning to automatically select the best kinetic model (f2–f8) for radiopharmaceutical 177Lu-PSMA each virtual data prostate cancer patient based on serial time–activity data. The system performs curve fitting, visualizes results, and provides mathematical formulas for pharmacokinetic analysis.

## Background
Internal radiation dosimetry is essential in nuclear medicine therapy, but manual model selection is slow and requires advanced expertise. Errors in dose calculation may lead to:
- Overdose: Excessive radiation → higher risk of side effects
- Underdose: Insufficient radiation → ineffective treatment

This system aims to:
- Automate dosimetry with machine learning
- Reduce time for complex calculations
- Reduce human resource burden
- Improve patient safety with precise, individualized dose calculations

> **Note:** IBM Granite AI was used **only during development** for code generation, optimization, and documentation. 

---

## Features

### Data Management
- Add Row: Add a new time–activity data point  
- Restart: Clear all data
- Predict: To predict the best model for the spesific time-activity data point  

### Automated Kinetic Model Prediction
- Supports models: f2–f8  
- Uses Random Forest Classifier to predict best fit  

### Curve Fitting & Parameter Estimation
- Displays fitted curve over raw data  
- Shows mathematical formula (LaTeX-rendered)  

### Results Visualization
- Graphical curve plot  
- Estimated kinetic parameters  

---

## Technologies Used

- **Frontend:** React.js (Bootstrap & KaTeX for formula rendering)  
- **Backend:** FastAPI (Python)  
- **Machine Learning:** Random Forest Classifier  
- **Visualization:** Matplotlib (Python), KaTeX (React)  
- **Data Handling:** Pandas, NumPy  

---

## Setup & Running Instructions (Local Development)

### 1. Clone Repository
```bash
git clone https://github.com/annrachman29/nuclear-medicine-dosimetry-model.git
cd nuclear-medicine-dosimetry-model
```

### 2. Backend Setup
Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```
Run backend server:
```bash
python final_app.py
```
The backend API will run on: http://127.0.0.1:8000


### 3. Frontend Setup
Install dependencies:
```bash
cd frontend
npm install
```
Run frontend:
```bash
npm start
```
The frontend development mode will run on (http://localhost:3000) in your browser.
> The page will reload when you make changes.


### 4. Using Virtual Patient Data
You can manually input data in the frontend interface for testing

**Example Input for Sample Data to Test the Application:**

| Time (h) | %ID/gr       |
|----------|--------------|
| 2        | 1.460908772  |
| 18       | 0.793761054  |
| 42       | 0.467941345  |
| 88       | 0.354520415  |
| 160      | 0.000581826  |

**Real Best Model Output: f2**

---

## AI Support Explanation
IBM Granite AI assisted during development to:
- Accelerate initial code generation (React, FastAPI, ML integration)
- Suggest UI/UX improvements (layout, responsive design)
- Optimize machine learning pipeline for faster prediction

---

## Recommended Further Development (After Research-Trial):
- Frontend: Vercel (for React apps)
- Backend: Render or Railway (supports FastAPI)
- Alternative: Firebase Hosting (if backend integrated as cloud function)

---

## Disclaimer
This project is for research and educational purposes only.\
It is not a certified medical device and must not be used in clinical settings without proper regulatory approval and validation.

---

## Author Affiliate Institution
- Nuclear Medicine Research Group, Physics Department, Faculty of Mathematics and Natural Science, Universitas Indonesia
- Indonesia's Nuclear Research Agency, National Research and Innovation Agency (BRIN)
- Medical Physics Program, Department of Radiology and Radiological Science & Department of Radiation Oncology and Molecular Radiation Sciences, School of Medicine, Johns Hopkins University


