from flask import Flask, render_template, request, redirect, url_for, session
import os
import pickle

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure random secret key

# Load model and vectorizer
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'svm_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# Simple user database
users = {'admin': '1234'}

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users.get(username) == password:
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return "Invalid Credentials"
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users[username] = password
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    prediction = None

    # Initialize session history if not already
    if 'history' not in session:
        session['history'] = []  # Store past predictions

    if request.method == 'POST':
        news_text = request.form['news']

        # Vectorize and predict
        vector = vectorizer.transform([news_text])
        pred = model.predict(vector)[0]
        prediction = "Real" if pred == 1 else "Fake"

        # Debugging: Print the new prediction
        print(f"Predicted: {prediction}, News: {news_text}")

        # Save to session history
        session['history'].append({'text': news_text, 'result': prediction})

        # Debugging: Print the updated session history
        print(f"Updated Session History: {session['history']}")

        # Ensure session changes are persisted
        session.modified = True  # This forces Flask to save the session data

    # Extract the labels and values for plotting the trend
    labels = [entry['text'] for entry in session['history']]
    values = [1 if entry['result'].lower() == 'fake' else 0 for entry in session['history']]
    colors = ['rgba(255, 99, 132, 0.5)' if val == 1 else 'rgba(75, 192, 192, 0.5)' for val in values]

    real_count = values.count(0)
    fake_count = values.count(1)

    # Debugging: Print the trend data
    print(f"Labels: {labels}")
    print(f"Values: {values}")
    print(f"Colors: {colors}")

    return render_template(
        'dashboard.html',
        prediction=prediction,
        entries=session['history'],
        labels=labels,
        values=values,
        colors=colors,
        real_count=real_count,
        fake_count=fake_count
    )

@app.route('/logout')
def logout():
    session.clear()  # Clear session when logging out
    return redirect(url_for('login'))

# For Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
