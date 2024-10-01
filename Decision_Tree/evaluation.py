from typing import List

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """
    precision =0
    recall =0
    if (len (expected_results) ==0 or len(actual_results) ==0 or len(actual_results) != len(expected_results) ):
        return None,None
    TP=0
    TN=0
    FP=0
    FN=0
    
    for i in range(len(expected_results)) :
        if(expected_results[i] == 1) :
            if (actual_results[i] == 1):
                TP=TP+1
            if (actual_results[i] == 0):
                FN=FN+1
        if(expected_results[i] == 0) :
            if (actual_results[i] == 1):
                FP=FP+1
            if (actual_results[i] == 0):
                TN=TN+1
    #print("TP = ",TP," FP = ",FP)       
    if(TP+FP == 0): 
        precision =0
    else :
        precision = TP/(TP+FP)
    if(TP+FN == 0):
        recall = 0
    else :
        recall = TP/(TP+FN)

    return precision,recall
    #raise NotImplementedError('Implement this method for Question 3')

def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    F1_score =0
    
    p,r = precision_recall(expected_results, actual_results)
    if (p+r == 0):
        F1_score = 0
    else :
        F1_score = 2*r*p/(r+p)
    
    #print("expected : ",expected_results," found :",actual_results,"precision : ",p," recall : ",r," score= ",F1_score)
    #print("precision : ",p," recall : ",r," score= ",F1_score)
    
    return F1_score

    # raise NotImplementedError('Implement this method for Question 3')
