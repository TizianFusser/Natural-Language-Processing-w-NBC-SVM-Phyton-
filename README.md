# Natural-Language-Processing-w-NBC-SVM-Phyton-
Classify reviews of prescripted medications on different disease w/ NLP. Testing the performance of Support Vector Classifier and Naive Bayse Classifier.

Business Problem

The dataset contains medical data like disease, medication, and a review of the medicines in text data. There is a binary variable 'Rating', which shows if the drug is rated high or low. Rating is going to be our target variable. We analyze the text data from reviews to give a classification of the review and the medication if it is highly rated or low rated.
Data preparation
The column rating was converted from categorical to numerical values. Two columns are dropped Medicine & Condition and tokenize, lemmatize and vectorize the text data.
The dataset contains 23304 rows with 5406 with the rating low and 17897 with the rating high. This is a 67/33 ratio; the dataset is imbalanced. We are not balancing the training set because we are using the support vector classifiers and naive Bayes, which can handle imbalanced datasets. We don't need to consider it due to the module is build using cross-validation and support vector classifiers / naive Bayes classifiers, which can handle slightly imbalanced data sets. 
Model Evaluation

The TF-IDF vectorizer uses a min_df=30, a  word needs to be at least 30 times in the dataset, and this parameter can be tuned with the grid search function.
Naive Bayes falls into the class of classifying generative structures. It models the posterior likelihood of conditional densities in the sample. So production is the probability of belonging to a class. 
Performance: Naive Bayes Classifier
List of cross-validation accuracies for NBC using uni-bi-trigrams:

[0.8335478335478336, 0.8241098241098241, 0.8532188841201717, 0.8339055793991417, 0.8373390557939914, 0.8167381974248927, 0.8291845493562232, 0.8351931330472103, 0.8257510729613734, 0.8253218884120171]

Mean cross-validation accuracy for NBC using uni-bi-trigrams:  0.8314310018172678
Best cross-validation accuracy for NBC using uni-bi-trigrams:  0.8532188841201717

In comparison, SVM is based on a discrimination function. Here the training data calculates the weights and the bias parameter. It tries to find a hyperplane that maximizes the margin and the function of optimization in this context. It works well with linear and non-linear problems. In our case, we are using the linear kernel.
Performance: Support Vector Classifier
List of cross-validation accuracies for SVC using uni-bi-trigrams: 
[0.8734448734448734, 0.8815958815958816, 0.8909871244635194, 0.8849785407725322, 0.8969957081545065, 0.8755364806866953, 0.8695278969957082, 0.8854077253218884, 0.8802575107296138, 0.8798283261802575]

Mean cross-validation accuracy for SVC using uni-bi-trigrams:  0.8818560068345477
Best cross-validation accuracy for SVC using uni-bi-trigrams:  0.8969957081545065

As we can see, the SVC outperforms the NBC, and SVC works better in imbalanced datasets than NBC because the algorithm doesn't take the whole data set into consideration, it creates the support vectors. It divides the classes by them with the hyperplane. It looks like the SVC works better with the high dimension dataset. The Navi Bayes classifier doesn't work so well with a dataset that has a higher dimension but is sparse, too. Because NBC calculates the probability of every word to be used for each class. More words result in higher dimensions because we are using the 'bag of words' technique.

Model deployment
We deploy the support vector classifier on a dataset with reviews without ratings and predicting the rating of the review' high' or 'low.' The dataset contains 853 rows. We predicted '603' with the rating 'high' and 249 with a rating 'low.'  This is still a 71/29 ratio.
The TF-IDF vectorizer uses a min_df=5. A word needs to be at least five times in the dataset, using it because the NoRatings data set is significantly smaller.
