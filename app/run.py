from flask import Flask, render_template, request, redirect, url_for, session
import os
import pickle
import json

app = Flask(__name__)
app.secret_key = 'FakeNot'

# Load your model (which should be a pipeline: vectorizer + classifier)
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'svm_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

# Load model and vectorizer
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# Check if history is stored in session, if not initialize it
if 'history' not in session:
    session['history'] = []

# Home route
@app.route('/')
def home():
    return redirect(url_for('login'))

# Simple users dictionary
users = {'admin': '1234'}

# Login route
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

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Add a new user (simple registration logic)
        users[username] = password
        return redirect(url_for('login'))
    return render_template('register.html')

# Dashboard route
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    prediction = None
    if request.method == 'POST':
        news_text = request.form['news']

        # Transform the input news text using the vectorizer
        vector = vectorizer.transform([news_text])  # transform text to feature vector
        pred = model.predict(vector)[0]  # predict using the model

        # Predict and determine the result
        prediction = "Real" if pred == 1 else "Fake"

        # Store the entry with text and prediction result in session history
        session['history'].append({'text': news_text, 'result': prediction})

    # Prepare data for rendering the history and prediction trend
    labels = [entry['text'] for entry in session['history']]
    values = [1 if entry['result'].lower() == 'fake' else 0 for entry in session['history']]
    colors = ['rgba(255, 99, 132, 0.5)' if val == 1 else 'rgba(75, 192, 192, 0.5)' for val in values]

    # Prepare data for the trend (count of Real and Fake)
    real_count = values.count(0)
    fake_count = values.count(1)

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

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
