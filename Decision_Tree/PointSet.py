from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.best_split = -1   ## meaning get_best_gain method was not called yet 
        self.min_split = 1
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """
        gini_score= 0
        n=len(self.labels)
        p_c1=self.labels.sum()
        p_c2=n-p_c1
        p_c1=p_c1/n
        p_c2=p_c2/n
        gini_score = 1-(p_c1**2+p_c2**2)
        return gini_score 
        
        #raise NotImplementedError('Please implement this function for Question 1')

    def get_best_gain(self,min_split_points =1) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """
        self.best_split=None,None ## here if we found it -2 we know that this function returned None => all features are the same or there is a #problem
        nbr_samples = len(self.labels)
        if (nbr_samples == 0):
            return None,None
        nbr_features =len(self.features[0])
        if (nbr_features ==0):
            return None,None
        if(self.features.sum()==nbr_samples*nbr_features or self.features.sum()==0):
            return None,None
        best_gain =-1
        index =-1
        gini = self.get_gini()
        is_type=0
        category_split = None
        threshold_split= None
       # print(f"this is the gini {gini}")
        ###### defining intermediate functions 
        
        def compute_gini_child_i_not_cat(feature_i,labels,cat) :
            current_feature = labels[~(feature_i==cat)]
            C1= (current_feature == 1.0).sum()
            C2= (current_feature == 0.0).sum()
            C=C1+C2
            if (C==0 and C<min_split_points) :
                return None,None 
            #print("here",C,C1,C2)
            gini_child = 1-((C1/C)**2 + (C2/C)**2) #######
            return gini_child,C
            
            
            
        def compute_gini_child_i (feature_i , labels , value):
            #print("hi")
            current_feature = labels[feature_i==value]
            C1= (current_feature == 1.0).sum()
            C2= (current_feature == 0.0).sum()
            C=C1+C2
            if (C==0 or C<min_split_points) :
                return None,None 
            #print("here",C,C1,C2)
            gini_child = 1-((C1/C)**2 + (C2/C)**2) #######
            return gini_child,C
        # returns the value of gini_split when using the feature i
        def compute_gini_split_i(feature_i,lables):
            gini_N1,N1=compute_gini_child_i(feature_i,lables,1.0)
            gini_N2,N2=compute_gini_child_i(feature_i,lables,0.0)
            if(N1 == None or N2==None  ) :
                return None
            if(N2<min_split_points or N1<min_split_points) :
                return None
            N=N1+N2
    
            gini_split = N1/N * gini_N1 + N2/N * gini_N2
            return gini_split

        def compute_gini_split_i_classes (feature_i,lables):
                categories=np.unique(feature_i)
                #print('Here you are my man categories are :',categories)
                n_categories = len(categories)
                
                if(n_categories ==1): ### all the instances are from the same category 
                    return None,None
                gini_splits =[]
                for cat_ind in range(n_categories):
                    cat = categories [cat_ind]
                    gini_N1,N1=compute_gini_child_i(feature_i,lables,cat)
                    gini_N2,N2=compute_gini_child_i_not_cat(feature_i,lables,cat)
                    if(N1 == None or N2==None ) :
                        return None,None
                    
                    if( N1<min_split_points or N2<min_split_points) :
                            return None,None
                    
                    N=N1+N2
            
                    gini_split = N1/N * gini_N1 + N2/N * gini_N2
                    gini_splits.append((gini_split,cat))
                    #print("cat = ",cat," gini = ",gini_split)
                best_gini,best_split = gini_splits[0]
                for gini_split,cat in gini_splits :
                    if(best_gini>gini_split):
                        best_gini,best_split = gini_split,cat
                
                return best_gini,best_split
            
        def compute_gini_split_i_real (feature_i,labels):
            
            candidates =np.unique(feature_i)
            n_candidates = len(candidates)
            #print("N = ",n_candidates)
            if(n_candidates ==1): ### all the instances have the same value
                    return None,None
            
    
            sorted_indices = np.argsort(feature_i)    
            sorted_feature_i=feature_i[sorted_indices]    
            sorted_labels = labels[sorted_indices] # sorted_according to the features 
            
            #print(f"here are the following\nfeatures_sorted {sorted_feature_i}\nlabels sorted {sorted_labels}")
            # starting the work 
            gini_split_min = 0.5   ## returned gini
            best_thresh = None      ## reterned threshold
            i = 0                ## i is the current index when iterating throw the feature_i values
            N1= 0                ## N1 is the number of samples less than the current threshold 
            N2= len(labels)      ## N2 is the number of samples greater than the current threshold
            N = N1+N2            ## number of instances of the data
            T1= 0                ## T1 is the number of samples in class 1 having True as label
            F1= 0                ## F1 is the number of samples in calss 1 having False as label
            T2= labels.sum()     ## T2 is the number of samples in calss 2 having True as label
            F2= N2-T2            ## F2 is the number of samples in calss 2 having False as label
            
            for thresh_index in range(len(candidates)-1) :
                thresh = candidates[thresh_index] ## current threshold
                N_I =0           ## number of the added values in classe 1 after changing the threshold
                T_I =0
                F_I =0
                while (sorted_feature_i[i] <= thresh and i<N)  :    ## iterate over the samples to count the number of added values 
                    N_I+=1
                    if(sorted_labels[i] == 1.0):
                        T_I +=1
                    else :
                        F_I +=1
                    i+=1  ### advancing in the samples
                
                ### now we found all what we need after changing the threshold let's compute the gini_split
                T1+= T_I
                T2-= T_I
                F1+= F_I
                F2-= F_I
                N1+= N_I
                N2-= N_I
                #print('this is the features :',feature_i,'\nthis is the candidates : ',candidates,'\nthis is the threshold',thresh,'this is N1 and N2',N1,N2)
                gini_1 = 1- (T1/N1)**2 -(F1/N1)**2
                gini_2 = 1- (T2/N2)**2 -(F2/N2)**2
                gini_split = (N1/N)*gini_1 + (N2/N)*gini_2
               # print(f"thresh : {thresh}, gini_split = {gini_split}, T1 : {T1} ,N1={N1} ,N2= {N2}T2={T2}")
                if( gini_split_min >gini_split and N1>=min_split_points and N2>=min_split_points) :
                    gini_split_min =gini_split
                    best_thresh =(thresh+candidates[thresh_index+1] )/2
            if(best_thresh == None):
                return None,None
            return gini_split_min,best_thresh
            
            
            
         ##### end definitions ######            
            
            
            
        for i in range(nbr_features) :  ### we iterate over all features
            feature_i=self.features[:,i]
            
         ####### case of boolean feature #######
            if(self.types[i]==FeaturesTypes.BOOLEAN):
                #print('nope not this time')
                gini_split = compute_gini_split_i(feature_i,self.labels)
               
                
                if(gini_split != None):
                    gini_gain = gini-gini_split
                    #print(f"here we are at bool feature {i} gini_split ={gini_split} gini_gain = {gini_gain}")
                    if (gini_gain>best_gain) :
                        best_gain = gini_gain
                        index = i 
                        is_type = 0
         ####### end case of boolean feature #######
            
           
            ###### case of categorical feature #######
            if(self.types[i]==FeaturesTypes.CLASSES) :
                gini_split,best_category_split = compute_gini_split_i_classes(feature_i,self.labels)
                if(gini_split != None):
                    gini_gain = gini-gini_split
                    #print(f"here we are at categroical feature {i} gini_split ={gini_split} gini_gain = {gini_gain} best category ={best_category_split}")
                
                    if (gini_gain>best_gain) :
                        best_gain = gini_gain
                        index = i 
                        is_type = 1
                        category_split = best_category_split 
            ####### end case of categorical feature ########
            
            
            
            
            ####### case of continous feature ########
            
            if(self.types[i]==FeaturesTypes.REAL) :
                gini_split,best_threshold_split = compute_gini_split_i_real(feature_i,self.labels)
                #print(f"here we are !!! : gini split : {gini_split}, best_threshold_split :{best_threshold_split}") 
                if(gini_split != None):
                    gini_gain = gini-gini_split
                    if (gini_gain>best_gain ) :
                           # print("this is working")
                            best_gain = gini_gain
                            index = i 
                            is_type = 2
                            threshold_split = best_threshold_split 
            
            ####### end case of continous feature ########
            
        #    print ("gini split is ",gini_split)
        #if(index ==-1):
         #   print("$$$$$$$$$$$$$$$best split is None")
       # else :
          #  print("$$$$$$$$$$$$$$$best split is ",gini-best_gain)
        ## case boolean    
        
        if (is_type ==0):
            if (index==-1):
                return None,None
            self.best_split = index,best_gain
            return index,best_gain
                
        ## case categories    
        if (is_type ==1):
            if (index==-1):
                return None,None,None
            self.best_split =index,category_split,best_gain 
            return index,best_gain
        
        ## case continuous
        
        if (is_type ==2):
            if (index==-1):
                return None,None,None
            self.best_split =index,threshold_split,best_gain 
            #return index,threshold_split,best_gain 
            return index,best_gain 
        
    def get_best_threshold(self) -> float :
        
        best_split = self.best_split
        if(best_split == -1):
            raise NotImplementedError('This functions needs to be called after using get_best_gain method, Please check you did that')
        elif (best_split == (None,None)):
            print("all values are the same and we couldn't find any splitting for this data")
        else :
            if(self.types[best_split[0]]  == FeaturesTypes.BOOLEAN ) : 
                return None
            
            if(self.types[best_split[0]]  == FeaturesTypes.CLASSES ) : 
                return best_split[1]
            
            if(self.types[best_split[0]]  == FeaturesTypes.REAL ) : 
                return best_split[1]
            

