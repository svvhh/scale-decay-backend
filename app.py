from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow calls from any GPT

@app.route("/v1/regression/scale-decay", methods=["POST"])
def run_regression():
    try:
        content = request.json
        data = content.get("data", [])
        metric = content.get("returnMetric", "moic")

        if not data or metric not in ["moic", "irr"]:
            return jsonify({"error": "Invalid input"}), 400

        df = pd.DataFrame(data)
        if "investmentSize" not in df.columns or metric not in df.columns:
            return jsonify({"error": "Missing required fields"}), 400

        df = df.dropna(subset=["investmentSize", metric])
        df = df[df["investmentSize"] > 0]
        df["logSize"] = np.log(df["investmentSize"])

        X = sm.add_constant(df["logSize"])
        y = df[metric]
        model = sm.OLS(y, X).fit()

        return jsonify({
            "coefficient": round(model.params["logSize"], 4),
            "intercept": round(model.params["const"], 4),
            "rSquared": round(model.rsquared, 4),
            "pValue": round(model.pvalues["logSize"], 4),
            "interpretation": (
                "There is statistically significant evidence of scale decay." if model.pvalues["logSize"] < 0.05
                else "There is no statistically significant evidence of scale decay."
            )
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
