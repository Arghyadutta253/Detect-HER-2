from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np

# Load model
model = pickle.load(open("breast_cancer_detector.pickle", "rb"))

app = Flask(__name__)
app.secret_key = "your_secret_key"  # required for session handling

# Dummy credentials (later you can connect to DB)
USERS = {
    "admin@example.com": "1234"
}

# ---------- LANDING PAGE ----------
@app.route("/")
def landing():
    return render_template("land2.html")

# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if email in USERS and USERS[email] == password:
            session["user"] = email
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid Email or Password!")

    return render_template("login.html")

# ---------- REGISTER ----------
@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")

    if email not in USERS:
        USERS[email] = password
        session["user"] = email
        return redirect(url_for("index"))
    else:
        return render_template("login.html", error="User already exists!")

# ---------- DASHBOARD / INDEX ----------
@app.route("/index")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# ---------- PREDICTION ----------
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        features = [
            float(request.form["mean_radius"]),
            float(request.form["mean_texture"]),
            float(request.form["mean_perimeter"]),
            float(request.form["mean_area"]),
            float(request.form["mean_smoothness"]),
            float(request.form["mean_compactness"]),
            float(request.form["mean_concavity"]),
            float(request.form["mean_concave_points"]),
            float(request.form["mean_symmetry"]),
            float(request.form["mean_fractal_dimension"]),
            float(request.form["radius_error"]),
            float(request.form["texture_error"]),
            float(request.form["perimeter_error"]),
            float(request.form["area_error"]),
            float(request.form["smoothness_error"]),
            float(request.form["compactness_error"]),
            float(request.form["concavity_error"]),
            float(request.form["concave_points_error"]),
            float(request.form["symmetry_error"]),
            float(request.form["fractal_dimension_error"]),
            float(request.form["worst_radius"]),
            float(request.form["worst_texture"]),
            float(request.form["worst_perimeter"]),
            float(request.form["worst_area"]),
            float(request.form["worst_smoothness"]),
            float(request.form["worst_compactness"]),
            float(request.form["worst_concavity"]),
            float(request.form["worst_concave_points"]),
            float(request.form["worst_symmetry"]),
            float(request.form["worst_fractal_dimension"])
        ]

        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)[0]
        result = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-Cancerous)"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
