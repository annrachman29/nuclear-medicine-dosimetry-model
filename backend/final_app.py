import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit

app = Flask(__name__)
CORS(app)

# --------------------------
# Mathematical models
# --------------------------
def safe_exp(x):
    x = np.clip(x, -700, 700)
    return np.exp(x)

def f2(t, A1, l1, lphys):
    return A1 * safe_exp(-(l1 + lphys) * t)

def f3(t, A1, l1, l2, lphys):
    return A1 * safe_exp(-(l1 + lphys) * t) - A1 * safe_exp(-(l2 + lphys) * t)

def f4(t, A1, A2, l1, l2, lbc, lphys):
    return A1*safe_exp(-(l1+lphys)*t) - A2*safe_exp(-(l2+lphys)*t) - (A1-A2)*safe_exp(-(lbc+lphys)*t)

def f5(t, A1, A2, l1, l2, l3, lphys):
    return A1*safe_exp(-(l1+lphys)*t) + A2*safe_exp(-(l2+lphys)*t) - (A1+A2)*safe_exp(-(l3+lphys)*t)

def f6(t, A1, A2, A3, l1, l2, l3, lbc, lphys):
    return A1*safe_exp(-(l1+lphys)*t) + A2*safe_exp(-(l2+lphys)*t) - A3*safe_exp(-(l3+lphys)*t) - (A1+A2-A3)*safe_exp(-(lbc+lphys)*t)

def f7(t, A1, A2, A3, l1, l2, l3, l4, lphys):
    return A1*safe_exp(-(l1+lphys)*t) + A2*safe_exp(-(l2+lphys)*t) - A3*safe_exp(-(l3+lphys)*t) - (A1+A2-A3)*safe_exp(-(l4+lphys)*t)

def f8(t, A1, A2, A3, A4, l1, l2, l3, l4, lbc, lphys):
    return A1*safe_exp(-(l1+lphys)*t) + A2*safe_exp(-(l2+lphys)*t) + A3*safe_exp(-(l3+lphys)*t) - A4*safe_exp(-(l4+lphys)*t) - (A1+A2+A3-A4)*safe_exp(-(lbc+lphys)*t)

models = {"f2": f2, "f3": f3, "f4": f4, "f5": f5, "f6": f6, "f7": f7, "f8": f8}

model_formulas = {
    "f2": r"f_2(t) = A_1 e^{-(\lambda_1 + \lambda_{phys}) t}",
    "f3": r"f_3(t) = A_1 e^{-(\lambda_1 + \lambda_{phys}) t} - A_1 e^{-(\lambda_2 + \lambda_{phys}) t}",
    "f4": r"f_4(t) = A_1 e^{-(\lambda_1 + \lambda_{phys}) t} - A_2 e^{-(\lambda_2 + \lambda_{phys}) t} - (A_1 - A_2) e^{-(\lambda_{bc} + \lambda_{phys}) t}",
    "f5": r"f_5(t) = A_1 e^{-(\lambda_1 + \lambda_{phys}) t} + A_2 e^{-(\lambda_2 + \lambda_{phys}) t} - (A_1 + A_2) e^{-(\lambda_3 + \lambda_{phys}) t}",
    "f6": r"f_6(t) = A_1 e^{-(\lambda_1 + \lambda_{phys}) t} + A_2 e^{-(\lambda_2 + \lambda_{phys}) t} - A_3 e^{-(\lambda_3 + \lambda_{phys}) t} - (A_1 + A_2 - A_3) e^{-(\lambda_{bc} + \lambda_{phys}) t}",
    "f7": r"f_7(t) = A_1 e^{-(\lambda_1 + \lambda_{phys}) t} + A_2 e^{-(\lambda_2 + \lambda_{phys}) t} - A_3 e^{-(\lambda_3 + \lambda_{phys}) t} - (A_1 + A_2 - A_3) e^{-(\lambda_4 + \lambda_{phys}) t}",
    "f8": r"f_8(t) = A_1 e^{-(\lambda_1 + \lambda_{phys}) t} + A_2 e^{-(\lambda_2 + \lambda_{phys}) t} + A_3 e^{-(\lambda_3 + \lambda_{phys}) t} - A_4 e^{-(\lambda_4 + \lambda_{phys}) t} - (A_1 + A_2 + A_3 - A_4) e^{-(\lambda_{bc} + \lambda_{phys}) t}"
}

# --------------------------
# Load CSV & train RandomForest
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder tempat final_app.py
csv_path = os.path.join(BASE_DIR, "data", "Combined_Virtual_Data_UniqueID.csv")
df = pd.read_csv(csv_path, sep=';')

valid_models = ['f2', 'f3', 'f4b', 'f5a', 'f6a', 'f7b', 'f8a']
df = df[df['Type'].isin(valid_models)]
df['Type'] = df['Type'].str.extract(r'(f\d+)')
df = df.sort_values(by=['ID', 'Time'])

X = df.groupby('ID').apply(lambda g: [[a, t] for a, t in zip(g['A_sim'], g['Time'])]).tolist()
y = df.drop_duplicates('ID')['Type'].tolist()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_flat = [sum(x, []) for x in X_train]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_flat, y_train)
print("✅ RandomForest model trained at startup")

# --------------------------
# Flask endpoints
# --------------------------
@app.route("/")
def home():
    return jsonify({"message": "Nuclear Medicine Dosimetry Modeller API running!"})

@app.route("/predict_model", methods=["POST"])
def predict_model():
    json_data = request.get_json()
    if not json_data or "data" not in json_data:
        return jsonify({"error": "No data provided"}), 400

    df_input = pd.DataFrame(json_data["data"])
    if df_input.empty:
        return jsonify({"error": "Input data is empty"}), 400

    try:
        X_input = [sum([[row["%ID/gr"], row["Time"]] for _, row in df_input.iterrows()], [])]
        pred_model = clf.predict(X_input)[0]

        if pred_model not in models:
            return jsonify({"error": f"Predicted model '{pred_model}' not implemented"}), 400

        model_func = models[pred_model]
        t = df_input["Time"].to_numpy()
        y = df_input["%ID/gr"].to_numpy()
        p0 = [1]*(len(model_func.__code__.co_varnames)-1)

        try:
            popt, _ = curve_fit(model_func, t, y, p0=p0, maxfev=5000)
            params = {k: float(v) for k, v in zip(model_func.__code__.co_varnames[1:len(popt)+1], popt)}
        except Exception as e:
            popt = None
            params = {}
            print("⚠️ Curve fit failed:", e)

        plt.figure(figsize=(6,4))
        plt.scatter(t, y, color="blue", label="Data")
        if popt is not None:
            t_fit = np.linspace(min(t), max(t), 100)
            y_fit = model_func(t_fit, *popt)
            plt.plot(t_fit, y_fit, "k--", label=f"Fit ({pred_model})")
        plt.xlabel("Time (h)")
        plt.ylabel("%ID/gr")
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return jsonify({
            "best_model": pred_model,
            "params": params,
            "formula": model_formulas.get(pred_model, "Formula not available"),
            "plot": img_base64
        })

    except Exception as e:
        print("DEBUG: Base64 length:", len(img_base64))
        return jsonify({"error": str(e)}), 500

# --------------------------
# Run Flask (Railway compatible)
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

