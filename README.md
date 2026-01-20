# Deep-learning-Sequential-model-.
#Using Python code to predict whether women ever had terminated pregnancy in a dataset from Demographic and health survey

This project implemented a deep learning model to predict whether a woman had ever experienced a terminated pregnancy using the 2018 Nigeria Demographic and Health Survey dataset.

The dataset contained 104,808 records. The dependent variable (column 1) was binary: 0 = never terminated pregnancy, 1 = has terminated pregnancy. Independent variables (columns 2–5) included respondent’s weight, wealth index, household size, and number of children ever born.

The data were shuffled and split into 80% training and 20% testing sets. Features were normalized before model training. A Sequential neural network with ReLU-activated hidden layers, dropout regularization, and a sigmoid output layer was trained using the Adam optimizer and binary cross-entropy loss. Model performance was evaluated using accuracy and a confusion matrix. Finally, the trained model was saved for future use.

The model achieved a training accuracy of 84.8% and a validation accuracy of 84.8%. On the test set, the confusion matrix showed 17,807 true negatives and 3,155 false negatives, indicating that the model predicted  cases as “no terminated pregnancy,” highlighting class imbalance and limited sensitivity to positive cases.


