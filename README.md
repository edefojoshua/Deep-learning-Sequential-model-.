# Deep-learning-Sequential-model-.
#Using Python code to predict whether women ever had terminated pregnancy in a dataset from Demographic and health survey

This project implemented a deep learning model to predict whether a woman had ever experienced a terminated pregnancy using the 2018 Nigeria Demographic and Health Survey dataset.

The dataset contained 104,808 records. The dependent variable (column 1) was binary: 0 = never terminated pregnancy, 1 = has terminated pregnancy. Independent variables (columns 2–5) included respondent’s weight, wealth index, household size, and number of children ever born.

The data were shuffled and split into 80% training and 20% testing sets. Features were normalized before model training. A Sequential neural network with ReLU-activated hidden layers, dropout regularization, and a sigmoid output layer was trained using the Adam optimizer and binary cross-entropy loss. Model performance was evaluated using accuracy and a confusion matrix. Finally, the trained model was saved for future use.

The model achieved a training accuracy of 84.8% and a validation accuracy of 84.8%. On the test set. The confusion matrix of the third commit showed a specificity of approximately 92.0% (16,326 correctly classified non-terminated pregnancies out of 17,743), while sensitivity remained low at approximately 5.7% (184 correctly identified terminated pregnancies out of 3,219), indicating that although the model effectively identified women who did not terminate a pregnancy, its ability to detect terminated pregnancies was limited, reflecting the constrained predictive capacity of the available covariates



