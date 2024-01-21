Titanic 

Python Library that are used in this project:
->Numpy
->Pandas
->Tensorflow
->matplotlib
->scikit-learn
->scipy
->dill
->django

METHODOLOGYs

**1.Problem Statement**

The information about the people who were in the titanic is given and you have to predict whether the person was lived or not in titanic accident using different machine learning algorithms.

**2.Data Collection**

Collected the data form kaggle. 
**Source:** https://www.kaggle.com/competitions/titanic/data

**3.Data Pre-Processing**

Collected data is loaded in ingestion.py file and it is converted to pandas Data Frame which is stored into the aritfacts folder.
Loaded the train and testing data from artifacts folder and remove unnecessary parameter which wont affect our model like(name of person, ticket number, cabin number etc).
Then the unfilled data are filled using sklearn.imputes.SimpleImputer():
For Categorical parameter the strategy is most frequest.
For Numerical parameter the strategy is mean value.
Numerical value are standardized using sklearn.preprocessing.StandardScalar().
Categorical value are converted into the ont hot encoded value using sklearn.preprocessing.OneHotEncoder().
Finally, the transformer is proceed to fit the data and it is store into the artifacts foder as prepocessor.pkl file.

**4.Model training**

For model training, I have used different machine learning algorithms like logistic regression, SVM, Naive Bayes, KNN, K Mean clustering, Decision Tree and Random Forest.
I have used concept of hyperparameter to find the best params so that our model will have higher efficeincy.
Using hyperparametr concept the best model is found using sklearn.model_selection.GridSearchCV and it is saved to artifacts folder as model.pkl file.

**5.Website creation**

I have created a website using django framework which will take the necessary inputs for our model and it will pass those details into prediciton pipelines so that it can predict the living chance of that person by loading model.pkl and preprocessor.pkl file from artifacts folder.

**Description of each file and folder:**

**1.artifacts**\
this folder contains the training and testing data which is used into out model.In addition , this folder contains the pkl file like preprocessor and pre trained model.

**2.logs**\
this folder contains all the logging information that your have logged duirng execution of my project

**3.src**\
    this folder contains all the files that are required for our model to train\
    ->components\
       >>>data_ingestion.py\
            data_ingestionConfig() method contains the path of train and test dataset\
            data_ingestion_initate() method will load the train and test data from the source and store it into the aritfacts folder\
       >>> data_transformation.py\
            data_transformationConfig() method will contain the path to store our transformer/preprocessor pickel file\
            data_transformaton_initate() method will transform our train and test data into the array that can be fed into our machine learning model and it will save my transformer/preprocessor into artifact folder as preprocessor.pkl\
        ->model_trainer.py\
            >>>model_trainerConfig() method have the path to store the trained model\
        `    >>>model_trainer_initate() method will train our model and it will save trained model into artifacts folder as model.pkl file\
    ->pipeline.predict_pipeline\
            >>>prediction() method will predict person who gave his/her data would survive or not if he was in the titanic event\
    ->__init__.py\
        this file is empty file but this file is used to represent that its parent folder can be used as a module \
    ->exception.py\
        I have created a customException which will print the filename, lineno and message of exception if any error occurs during the execution\
    ->logger.py\
        This file contains the logging basic Configuration about how to store the log information in log folder\
    ->utils.py\
        this file is used to store such a method that is being used in most of the case during the execution of my project.\


**4.website**\
    this is a django project that contain a app name 'app' which contain all the information about the frontend of the website.\
    ->app\
        >>>migration\
            this folder contains all the information about the change in our models.py \
        >>>templates\
            this folder contains all the information about the html file for our website\
        >>>models.py\
            this file contains the structure of our table that we have to make\
        >>>views.py\
            this file handel the HTTP request\
    ->website\
        >>>urls.py\
            this file contain the url of each method defined in views.py\
    ->db.sqlite3\
        it is the database of our project\
    ->manage.py\
        this file contains all the information about commands that we write in terminal like makemigrations, migrates ,runserver etc\

**5.gitignore**\
this file contains the extension of such file which we dont want to push to our github\

**6.requirements.txt**\
this file contains the required python library for our project.
For example: tensorflow, matplotlib, numpy, pandas, keras, django etc.\


**7.setup.py**\
this file is used to setup our src file and it help to convert our src file to python module so that we can use it for other work.\

