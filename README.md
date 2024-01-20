Titanic 

Python Library that are used in this project:
1.Numpy
2.Pandas
3.Tensorflow

METHODOLOGYs

1.Problem Statement
The information about the people who were in the titanic is given and you have to predict whether the person was lived or not in titanic accident using different machine learning algorithms.

2.Data Collection
Collected the data form kaggle. 
Source: https://www.kaggle.com/competitions/titanic/data

3.Data Pre-Processing
Collected data is loaded in ingestion.py file and it is converted to pandas Data Frame which is stored into the aritfacts folder.
Loaded the train and testing data from artifacts folder and remove unnecessary parameter which wont affect our model like(name of person, ticket number, cabin number etc).
Then the unfilled data are filled using sklearn.imputes.SimpleImputer():
For Categorical parameter the strategy is most frequest.
For Numerical parameter the strategy is mean value.
Numerical value are standardized using sklearn.preprocessing.StandardScalar().
Categorical value are converted into the ont hot encoded value using sklearn.preprocessing.OneHotEncoder().
Finally, the transformer is proceed to fit the data and it is store into the artifacts foder as prepocessor.pkl file.

4.Model training
For model training, I have used different machine learning algorithms like logistic regression, SVM, Naive Bayes, KNN, K Mean clustering, Decision Tree and Random Forest.
I have used concept of hyperparameter to find the best params so that our model will have higher efficeincy.
Using hyperparametr concept the best model is found using sklearn.model_selection.GridSearchCV and it is saved to artifacts folder as model.pkl file.

5.Website creation
I have created a website using django framework which will take the necessary inputs for our model and it will pass those details into prediciton pipelines so that it can predict the living chance of that person by loading model.pkl and preprocessor.pkl file from artifacts folder.

