# function that takes a testing data set, a classification model, and returns the different scores for the testing set

def model_scores(X_test, y_test, model):
    
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    
    # predictions
    preds = model.predict(X_test)
    
    # confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    # scores
    accuracy = model.score(X_test, y_test)
    missclassification = 1 - accuracy
    specificity = tn/(tn+fp)
    recall_sensitivity = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = 2 * (precision*recall_sensitivity)/(precision+recall_sensitivity)
    
    # data frame
    scores_df = pd.DataFrame({
        'score' : ['accuracy', 'missclassification', 'specificity', 'recall_sensitivity', 'precision', 'f1'],
        'value': [accuracy, missclassification, specificity, recall_sensitivity, precision, f1]
    })
    
    return scores_df

