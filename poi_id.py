#!/usr/bin/python

### James Dellinger
### Udacity Intro to Machine Learning Course
### Final Project
### February 22, 2018


import sys
import pickle
import numpy
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### In Task 3 I will use univariate feature selection to whittle this list
### down to the most effective features. I am including the first feature that
### I engineered ("percentage_total_stock_exercised"), but not the second one
### ("ratio_of_exercised_stock_to_payments")
features_list = ['poi','deferral_payments','expenses','deferred_income','long_term_incentive','shared_receipt_with_poi','loan_advances','other','bonus','total_stock_value','from_poi_to_this_person','from_this_person_to_poi','restricted_stock','percentage_total_stock_exercised','salary','total_payments','exercised_stock_options']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Get the number of datapoints (individual entries) in the dataset
sizeOfDataset = len(data_dict)
print "Size of dataset (in individuals): ", sizeOfDataset

### Get the number of features each datapoint contains by default
numberOfFeatures = len(data_dict["LAY KENNETH L"])
print "Number of features for each entry in dataset, by default :", numberOfFeatures

### Get the number of POIs who are in the dataset
poiCount = 0
for key in data_dict:
    if data_dict[key]["poi"] == True:
        poiCount += 1
print "Number of POIs in dataset: ", poiCount

### Finally, get the number of indivuduals not previously flagged as a POI
print "Number of non-POIs in dataset", sizeOfDataset - poiCount

### Make sure no POIs have values for the "director_fees" and
### "restricted_stock_deferred" features
poiWithDirectorFeesCount = 0
poiWithRestrictedStockDeferredCount = 0
for key in data_dict:
    if data_dict[key]["poi"] == True and data_dict[key]["director_fees"] != "NaN":
        poiWithDirectorFeesCount += 1
    if data_dict[key]["poi"] == True and data_dict[key]["restricted_stock_deferred"] != "NaN":
        poiWithRestrictedStockDeferredCount += 1
print "Number of POIs with values for director_fees feature: ", poiWithDirectorFeesCount
print "Number of POIs with values for restricted_stock_deferred feature: ", poiWithRestrictedStockDeferredCount

### Task 2: Remove outliers
### Remove TOTAL outlier (included due to a spreadsheet quirk)
data_dict.pop("TOTAL",0)
### Also remove THE TRAVEL AGENCY IN THE PARK outlier
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)
print "Number of data points in dataset after removing outliers: ", len(data_dict)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### My first attempt to engineer a new feature: "percentage_total_stock_exercised"
### Calculate Exercised Stock Options as a percentage (as a decimal) of Total
### Stock Value for each name in the dataset dictionary. Store inside the
### data dictionary under the key "percentage_total_stock_exercised"
def engineerPercentageTotalStockExercisedFeature(dataset):
    for name in dataset:
        totalStockValue = dataset[name]["total_stock_value"]
        exercisedStockOptions = dataset[name]["exercised_stock_options"]
        ### Calculate percentage of total stock value exercised as options,
        ### or set to "NaN" if values for either "total_stock_value" or
        ### "exercised_stock_options" features are missing for a name.
        if totalStockValue != "NaN" and exercisedStockOptions != "NaN":
            percentageTotalStockExercised = exercisedStockOptions*1.0/totalStockValue
        else:
            percentageTotalStockExercised = "NaN"
        ### Add value for "percentage_total_stock_exercised" feature to the
        ### name's entry in the data dictionary.
        dataset[name]["percentage_total_stock_exercised"] = percentageTotalStockExercised
    return dataset

### Add this new feature to the data dictionary
my_dataset = engineerPercentageTotalStockExercisedFeature(my_dataset)

### Plot my newly engineered "percentage_total_stock_exercised" feature on the
### X-axis versus the "exercised_stock_options" feature on the Y-axis. Display
### dots representing POIs in red. This visualization will help me see if there ### is any promise in using this new feature in my algorithm.
def plotPercentageTotalStockExercisedFeature(dataset):
    test_features_list = ['poi','percentage_total_stock_exercised','exercised_stock_options']
    data = featureFormat(dataset, test_features_list, sort_keys = True)
    for point in data:
        percentageTotalStockExercised = point[1]
        exercisedStockOptions = point[2]
        ### Draw the point blue of the name hasn't been flagged as a POI,
        ### and red if the name is that of a known POI.
        if point[0] == 0:
            matplotlib.pyplot.scatter(percentageTotalStockExercised, exercisedStockOptions, color = "b")
        if point[0] == 1:
            matplotlib.pyplot.scatter(percentageTotalStockExercised, exercisedStockOptions, color = "r")
    matplotlib.pyplot.xlabel("Percentage of Total Stock Value Exercised")
    matplotlib.pyplot.ylabel("Dollar Value of Exercised Stock Options")
    matplotlib.pyplot.show()

plotPercentageTotalStockExercisedFeature(my_dataset)

### My second attempt to engineer a new feature: "ratio_of_exercised_stock_to_payments"
### Calculate the ratio of exercised stock options to total payments for each
### name in the dataset dictionary. Store inside the data dictionary under the
### key "atio_of_exercised_stock_to_payments"
def engineerRatioExercisedStockToTotalPaymentsFeature(dataset):
    for name in dataset:
        totalPayments = dataset[name]["total_payments"]
        exercisedStockOptions = dataset[name]["exercised_stock_options"]
        ### Calculate ratio of exercised stock options to total payments,
        ### or set to "NaN" if values for either "total_payments" or
        ### "exercised_stock_options" features are missing for a name.
        if totalPayments != "NaN" and exercisedStockOptions != "NaN":
            ratioExercisedStockToTotalPayments = exercisedStockOptions*1.0/totalPayments
        else:
            ratioExercisedStockToTotalPayments = "NaN"
        ### Add value for "percentage_total_stock_exercised" feature to the
        ### name's entry in the data dictionary.
        dataset[name]["ratio_of_exercised_stock_to_payments"] = ratioExercisedStockToTotalPayments
    return dataset

### Add this new feature to the data dictionary
my_dataset = engineerRatioExercisedStockToTotalPaymentsFeature(my_dataset)

### Plot my newly engineered "ratio_of_exercised_stock_to_payments" on the ### ### X-axis versus the "total_payments" feature on the Y-axis. Display dots
### representing POIs in red. This visualization will help me see if there is
### any promise in using this new feature in my algorithm.
def plotRatioExercisedStockToTotalPaymentsFeature(dataset):
    test_features_list = ['poi','ratio_of_exercised_stock_to_payments','total_payments']
    data = featureFormat(dataset, test_features_list, sort_keys = True)
    for point in data:
        ratioExercisedStockToTotalPayments = point[1]
        totalPayments = point[2]
        ### Draw the point blue of the name hasn't been flagged as a POI,
        ### and red if the name is that of a known POI.
        if point[0] == 0:
            matplotlib.pyplot.scatter(ratioExercisedStockToTotalPayments, totalPayments, color = "b")
        if point[0] == 1:
            matplotlib.pyplot.scatter(ratioExercisedStockToTotalPayments, totalPayments, color = "r")
    matplotlib.pyplot.xlabel("Ratio of Exercised Stock to Total Payments")
    matplotlib.pyplot.ylabel("Total Payments")
    matplotlib.pyplot.show()

plotRatioExercisedStockToTotalPaymentsFeature(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Use univariate feature selection (SelectKBest) to pick the highest scoring
### features from our dataset to use with our machine learning algorithm.
from sklearn.feature_selection import SelectKBest, f_classif
### Experimented with values as low as k=3 and as high as k=10. Using k=7
### seemed to consistently yield the best results when used with my Decision
### Tree classifier that I tuned with GridSearchCV below. It appears that k=7
### gives the best balance between bias and variance.
selector = SelectKBest(f_classif, k=7)
selector.fit(features, labels)
features_transformed = selector.transform(features)
### Display the 7 features that were chosen and print out their feature scores
selectorResultsArray = selector.get_support()
featureScores = selector.scores_
print "Seven highest-scoring features according SelectKBest: "
indexCounter = 0
for result in selectorResultsArray:
    if result == True:
        print features_list[indexCounter], "(feature score: ", featureScores[indexCounter], ")"
        indexCounter += 1
    else:
        indexCounter += 1

### Update features_list variable to reflect the seven features chosen by
### univariate feature selection using SelectKBest with k=7. (This will be
### dumped and then used by the testing script to evaluate my algorithm.)
features_list = ['poi','expenses','deferred_income','other','bonus','from_this_person_to_poi','percentage_total_stock_exercised','total_payments']

### Again, extract features and labels from dataset for tuning and testing the
### parameters of my classifier algorithms, this time using the features_list
### that I got from running SelectKBest.
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Segment datapoints into training and testing segments in order to
### validate chosen classifiers.
### Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### Instead, I am using KFold to cross-validate, train and test my algorithms.
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True)
for train_indices, test_indices in kf.split(features):
    features_train = [features[ii] for ii in train_indices]
    features_test = [features[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Will implement both a Decision Tree classifier and a Regression
### classifier, and determine which one does a better job.

### First, implement a Decision Tree:
from sklearn import tree
clfDecisionTree = tree.DecisionTreeClassifier()
clfDecisionTree = clfDecisionTree.fit(features_train, labels_train)
predDecisionTree = clfDecisionTree.predict(features_test, labels_test)

### Accuracy of the Decision Tree classifier
from sklearn.metrics import accuracy_score
accDecisionTree = accuracy_score(predDecisionTree, labels_test)
print "Accuracy of the Decision Tree classifier (un-tuned): ", accDecisionTree

### Next, try a Naive Bayes classifier:
from sklearn.naive_bayes import GaussianNB
clfNaiveBayes = GaussianNB()
clfNaiveBayes = clfNaiveBayes.fit(features_train, labels_train)
predNaiveBayes = clfNaiveBayes.predict(features_test)

### Accuracy of the Naive Bayes classifier
accNaiveBayes = clfNaiveBayes.score(features_test, labels_test)
print "Accuracy of the Naive Bayes classifier: ", accNaiveBayes

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Let's tune the parameters of our Decision Tree classifier using
### GridSearchCV
from sklearn.model_selection import GridSearchCV
decisionTree = tree.DecisionTreeClassifier()
parametersDecisionTree = {'criterion':('gini', 'entropy'), 'splitter':('best', 'random'), 'min_samples_split':[2,4,6,8]}
clfDecisionTreeWithGridSearchCV = GridSearchCV(decisionTree, parametersDecisionTree)
clfDecisionTreeWithGridSearchCV = clfDecisionTreeWithGridSearchCV.fit(features_train, labels_train)
predDecisionTreeWithGridSearchCV = clfDecisionTreeWithGridSearchCV.predict(features_test)

### Accuracy of the Decision Tree classifier tuned by GridSearchCV
accDecisionTreeWithGridSearchCV = accuracy_score(predDecisionTreeWithGridSearchCV, labels_test)
print "Accuracy of the Decision Tree classifier tuned by GridSearchCV: ", accDecisionTreeWithGridSearchCV

### Accuracy of my GridSearchCV-tuned Decision Tree classifier was
### consistently higher than that of both my un-tuned Decision Tree
### classifier, as well as that of my Naive Bayes classifier.
###
### The best parameters for the Decision Tree classifier after tuning with
### GridSearchCV:
bestParametersForDecisionTree = clfDecisionTreeWithGridSearchCV.best_params_
print "Best parameters for Decision Tree classifier after tuning with GridSearchCV: ", bestParametersForDecisionTree

### Tuning my Decision Tree classifier with GridSearchCV did not consistently
### result in the same "best parameters" returned after each attempt.
### Some patterns that I nonetheless noticed:
### 1. Lower values for min_samples_split more consistently yielded higher
###    accuracies
### 2. Similarly, on the whole, setting splitter to "best" and criterion to
###    "gini" also seemed to correlate more closely with better accuracies.
###
### After bit more guessing and checking, inspired by the above two guidelines,
### the final classifier that I ultimately chose was this one below. When
### running this classifier in tester.py, precision and recall were both above
### .3 on every single attempt.
clf = tree.DecisionTreeClassifier(min_samples_split=2, splitter='best', criterion='gini')

### The feature importances for my tuned Decision Tree classifier:
clf.fit(features_train, labels_train)
featureImportances = clf.feature_importances_
### Print out each feature's importance:
print "Feature importances for my tuned Decision Tree classifier (clf = tree.DecisionTreeClassifier(min_samples_split=2, splitter='best', criterion='gini')) :", featureImportances
indexCounter = 0
for importance in featureImportances:
    print features_list[indexCounter], "(Feature Importance score: ", featureImportances[indexCounter], ")"
    indexCounter += 1

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
