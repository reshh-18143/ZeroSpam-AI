# Spam Detection using NLP and Machine Learning

This project focuses on detecting spam messages using Natural Language Processing (NLP) techniques and machine learning models. The goal is to preprocess text data and classify messages as spam or not spam.

## Dependencies

To run this project, you need to have the following libraries installed:
- pandas
- numpy
- scikit-learn
- nltk
- pywin32
- tkinter

You can install these libraries using Python's package manager, pip.

## Dataset

The dataset should be loaded into a pandas DataFrame. It's important to preprocess the text data by removing stopwords, lemmatizing, and applying TF-IDF vectorization.

## Usage

1. **Preprocess Data**: 
    - Clean the text data.
    - Remove stopwords.
    - Lemmatize the words.

2. **TF-IDF Vectorization**: 
    - Convert the cleaned text data into TF-IDF vectors.

3. **Train-Test Split**: 
    - Split the dataset into training and testing sets.

4. **Train Model**: 
    - Train a machine learning model (e.g., Multinomial Naive Bayes) using the training data.

5. **Evaluate Model**: 
    - Evaluate the performance of the trained model on the test set.
    - Calculate the accuracy of the model.

6. **Save Model**: 
    - Save the trained model and the TF-IDF vectorizer for future use.

## Acknowledgments

- This project is built using open-source libraries such as pandas, numpy, scikit-learn, and nltk.
- Special thanks to the contributors, and the open-source community for their valuable resources and support.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
