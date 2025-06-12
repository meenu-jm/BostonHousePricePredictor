from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('boston_model.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from form and convert to float
        data = [float(x) for x in request.form.values()]
        final_input = np.array([data])

        # Make prediction
        prediction = model.predict(final_input)

        # Render result
        return render_template('index.html', prediction_text=f"Estimated House Price: ${round(prediction[0], 2)}K")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
