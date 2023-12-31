<h1> IMDb Sentiment Analysis with LSTM and TensorFlow.js Integration </h1>


This project implements a sentiment analysis model using LSTM (Long Short-Term Memory) neural networks to classify movie reviews from the IMDb dataset as positive or negative. The model is trained on a subset of the IMDb dataset, consisting of movie reviews labeled with sentiment polarity. Additionally, it includes integration with TensorFlow.js to export the trained model for web deployment.

<h2>Dataset</h2>

The IMDb dataset is a widely used benchmark dataset for sentiment analysis. It contains movie reviews along with their associated sentiment polarity (positive or negative). The dataset has been preprocessed and tokenized, and each review is represented as a sequence of word indices.

<h2>Model Architecture</h2>

The sentiment analysis model is built using a sequential neural network architecture in TensorFlow. The model architecture consists of the following layers:

  - Embedding Layer: This layer maps the integer-encoded word indices to dense vectors of fixed size. It learns to represent words in a continuous vector space, capturing semantic relationships between words.

  - Bidirectional LSTM Layer: This layer processes the sequence of word vectors in both forward and backward directions, allowing the model to capture context from past and future states. The LSTM layer has 100 units and includes dropout regularization to prevent overfitting.

  - Dense Layer: The final dense layer with a sigmoid activation function produces a binary classification output, indicating the sentiment polarity (positive or negative) of the input review.

<h2>Model Training</h2>

The model is trained using the compiled configuration, including the binary cross-entropy loss function and the Adam optimizer. The training data is split into training and validation sets. During training, the model's performance on the validation set is monitored using the EarlyStopping callback, which stops training if the validation loss doesn't improve after two consecutive epochs.

<h2>TensorFlow.js Integration</h2>

After training, the model is saved in the TensorFlow SavedModel format using the TensorFlow.js library. This allows the trained model to be exported and used in web applications for inference in JavaScript.

<h2>Model Evaluation</h2>

After training and TensorFlow.js export, the model's accuracy is evaluated on the test dataset to assess its performance. The accuracy metric indicates the percentage of correctly classified movie reviews.

<h2>Usage</h2>

Install the required dependencies (TensorFlow, NumPy, TensorFlow.js).

Download and preprocess the IMDb dataset using the provided imdb.load_data function.

Run the code to train the sentiment analysis model on the IMDb dataset and save it in the TensorFlow SavedModel format.

Evaluate the model's accuracy on the test dataset.

Use TensorFlow.js to load the exported model and perform inference in JavaScript web applications.

Requirements:
  - TensorFlow
  - NumPy
  - TensorFlow.js
  - IMDb dataset (automatically downloaded by imdb.load_data)

<h2>Conclusion</h2>
This project demonstrates the implementation of a sentiment analysis model using LSTM neural networks for classifying movie reviews. The model achieves accuracy on par with state-of-the-art methods for sentiment analysis tasks. Additionally, it includes integration with TensorFlow.js to export the trained model for web deployment.
