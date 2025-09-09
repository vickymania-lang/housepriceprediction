from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io, base64

app = FastAPI()

# Load trained model
model = joblib.load("house_price_model.pkl")

# Load dataset (make sure df has columns area, bedrooms, age, price)
df = pd.read_csv("homeprices.csv")  # replace with your dataset file

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>House Price Prediction</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container mt-5">
            <div class="card shadow-lg p-4 rounded-4">
                <h2 class="text-center text-primary mb-4">🏠 House Price Prediction</h2>
                <form action="/predict_form" method="post">
                    <div class="mb-3">
                        <label class="form-label">Area (sqft)</label>
                        <input type="number" class="form-control" name="area" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Bedrooms</label>
                        <input type="number" class="form-control" name="bedrooms" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Age (years)</label>
                        <input type="number" class="form-control" name="age" required>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Predict Price</button>
                </form>
            </div>
        </div>
    </body>
    </html>
    """

@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(area: float = Form(...), bedrooms: int = Form(...), age: int = Form(...)):
    # Prediction
    X_new = pd.DataFrame([[area, bedrooms, age]], columns=['area','bedrooms','age'])
    price = model.predict(X_new)[0]

    # Generate chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["area"], df["price"], color="red", marker="+", label="Dataset")
    ax.scatter(area, price, color="blue", marker="o", s=100, label="Your Prediction")
    ax.set_xlabel("Area (sqft)")
    ax.set_ylabel("Price")
    ax.legend()

    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prediction Result</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container mt-5">
            <div class="card shadow-lg p-4 rounded-4 text-center">
                <h2 class="text-success">✅ Prediction Result</h2>
                <p class="fs-4">Estimated House Price: <b>${price:,.2f}</b></p>
                <img src="data:image/png;base64,{image_base64}" class="img-fluid rounded mt-3" alt="Chart">
                <a href="/" class="btn btn-secondary mt-3">🔙 Back</a>
            </div>
        </div>
    </body>
    </html>
    """
