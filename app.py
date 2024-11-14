from flask import Flask, request, jsonify, render_template
import joblib

# Load the trained model and vectorizer
model = joblib.load('sms_spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML page


@app.route('/predict', methods=['POST'])
def predict():
    # Get the message from the POST request
    data = request.get_json(force=True)
    message = data['message']

    # Vectorize the message
    message_vectorized = vectorizer.transform([message])

    # Predict whether the message is spam or not
    prediction = model.predict(message_vectorized)

    # Return the prediction as JSON
    result = {'prediction': 'spam' if prediction[0] == 1 else 'ham'}
    return jsonify(result)


@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content


if __name__ == '__main__':
    app.run(debug=True)
