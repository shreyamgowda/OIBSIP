1. Import Libraries
First, the required Python libraries were imported. These libraries help with data handling, visualization, text processing, machine learning, and model evaluation.
2. Load the Dataset
The SMS Spam Collection dataset was loaded into the project. It contains two columns:
Label: Indicates whether the message is spam or ham (not spam).
Message: The actual SMS text.
3. Explore the Data
The dataset was examined to understand its structure, check for missing values, identify duplicate records, and observe the distribution of spam and ham messages.
4. Data Cleaning
The text messages were cleaned by:
Converting all text to lowercase.
Removing numbers and punctuation.
Removing common stop words (such as "the", "is", "and"). This makes the text more suitable for machine learning.
5. Data Visualization
A count plot was created to compare the number of spam and ham messages. A word cloud was also generated to visualize the most frequently occurring words in spam messages.
6. Feature Extraction
The cleaned text was converted into numerical features using the TF-IDF (Term Frequency–Inverse Document Frequency) technique so that machine learning models could process the text.
7. Split the Dataset
The dataset was divided into:
Training Data (80%) to train the models.
Testing Data (20%) to evaluate their performance.
8. Train Machine Learning Models
Two classification algorithms were trained:
Naive Bayes
Logistic Regression
9. Evaluate the Models
The trained models were evaluated using:
Accuracy
Precision
Recall
F1-Score
Classification Report
Confusion Matrix
These metrics helped measure how well the models identified spam and non-spam messages.
10. Predict New Messages
Finally, a new SMS message was provided to the trained model. Based on its content, the model predicted whether the message was Spam or Not Spam.