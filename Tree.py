from typing import List
import numpy as np
from PointSet import PointSet, FeaturesTypes

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
    """
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1,
                min_split_points : int =1 ):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """
        self.points = PointSet( features,labels, types)
        self.right_node=None  ###
        self.left_node= None ###
        self.h=h
        self.is_leaf = 0
        if(h>0):
           
        
        
        
            self.points.get_best_gain(min_split_points)
            
            self.indx_split_feature=self.points.best_split
            
            
      
         #   print(self.points.get_best_threshold())
            #print("\nhere the splitting index is ",self.indx_split_feature,'hight is h = ',h,'\n')
            
                
                
            
            
            
            #print(f"here we are {self.indx_split_feature}")
            if(self.indx_split_feature[0] != None):
                    #print("we are working")
                    ## here it makes sense to split
                   # print(self.points.features.shape,min_split_points,  self.indx_split_feature,self.points.types[self.points.best_split[0]])
                    features_=np.array(features)
                    labels_=np.array(labels)
                    #types_=np.array(types)
                    feature_i=features_[:,self.indx_split_feature[0]]

                    if(types[self.indx_split_feature[0]]==FeaturesTypes.BOOLEAN):
                        mask_1 = feature_i == 1.0
                        mask_0 = feature_i == 0.0
                    if(types[self.indx_split_feature[0]]==FeaturesTypes.CLASSES):
                        mask_1 = feature_i == self.indx_split_feature[1]
                        mask_0 = ~mask_1
                    if(types[self.indx_split_feature[0]]==FeaturesTypes.REAL):
                        mask_1 = feature_i <= self.indx_split_feature[1]
                        mask_0 = ~mask_1
                        
                        
                        
                    #print('here is the indx' ,self.indx_split_feature[0])
                    #print("the shape i ", feature_i.shape,"the shape of all ",features_.shape)
                    #print("here is the feature i",feature_i,"\nand the index ",self.indx_split_feature)
                    #print('here the mask',mask_0,'\nhere are the features',features_)
                    left_features = features_[mask_0].tolist() ####
                    left_labels = labels_[mask_0].tolist()
                    #left_types = types_[mask_0].tolist()
                    right_features = features_[mask_1].tolist()
                    right_labels = labels_[mask_1].tolist()
                    #right_types = types_[mask_1].tolist()
                    
                    nb_split_point_right = mask_1.sum()
                    nb_split_point_left = mask_0.sum()
                    if(nb_split_point_right>= min_split_points and  nb_split_point_left >=min_split_points):

                        if(h==1):
                            self.is_leaf = 1
                            self.left_node = PointSet(left_features,left_labels,types)
                            self.right_node = PointSet(right_features,right_labels,types)
                        else :

                            self.left_node = Tree(left_features,left_labels,types,h-1,min_split_points)
                        #print("constructed left successfully",self.h);
                            self.right_node = Tree(right_features,right_labels,types,h-1,min_split_points)
                        #print("constructed right successfully",self.h);
                    else : 
                        self.is_leaf = 2
            else :
                self.is_leaf = 2
                #print("here the splitting doesn't make sense ",self.indx_split_feature,'h=',h)
 #       raise NotImplementedError('Implement this method for Question 4')

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        predicted_label = None ;
    
        if (self.is_leaf==2): ## leaf because doesn't make sense to split more
            #print("\n whatever dude leaf is 2 ",self.is_leaf)
            n_samples = len(self.points.labels)
            n_c1 = self.points.labels.sum()
            n_c0 = n_samples-n_c1
            if(n_c1>n_c0 ):
                    return True
            else :
                    return False
        if(self.is_leaf==1): # is a leaf
            #print("\nwe are here h = 1\n")
            if(self.points.types[self.indx_split_feature[0]] == FeaturesTypes.BOOLEAN) : ## bolean type
                if(features[self.indx_split_feature[0]]==1):
                    n_samples = len(self.right_node.labels)
                    n_c1 = self.right_node.labels.sum()
                    n_c0 = n_samples-n_c1
                    if(n_c1>n_c0 ):
                        return True
                    else :
                        return False



                if(features[self.indx_split_feature[0]]==0):
              #      print("\nwe are here h = 1\n")
                #    print("\n whatever left_node.h= : ",self.left_node.h)
                    n_samples = len(self.left_node.labels)
                    n_c1 = self.left_node.labels.sum()
                    n_c0 = n_samples-n_c1
                    if(n_c1>n_c0 ):
                        return True
                    else :
                        return False
               
            
            if(self.points.types[self.indx_split_feature[0]] == FeaturesTypes.CLASSES) : ## classes type
                if(features[self.indx_split_feature[0]]==self.indx_split_feature[1]): ## same cat as the splitting
                    n_samples = len(self.right_node.labels)
                    n_c1 = self.right_node.labels.sum()
                    n_c0 = n_samples-n_c1
                    if(n_c1>n_c0 ):
                        return True
                    else :
                        return False



                if(features[self.indx_split_feature[0]]!=self.indx_split_feature[1]): # not the same
              #      print("\nwe are here h = 1\n")
                #    print("\n whatever left_node.h= : ",self.left_node.h)
                    n_samples = len(self.left_node.labels)
                    n_c1 = self.left_node.labels.sum()
                    n_c0 = n_samples-n_c1
                    if(n_c1>n_c0 ):
                        return True
                    else :
                        return False
                
                
            if(self.points.types[self.indx_split_feature[0]] == FeaturesTypes.REAL) : ### real type
               
                if(features[self.indx_split_feature[0]]<=self.indx_split_feature[1]): ## same cat as the splitting
                    n_samples = len(self.right_node.labels)
                    n_c1 = self.right_node.labels.sum()
                    n_c0 = n_samples-n_c1
                    if(n_c1>n_c0 ):
                        return True
                    else :
                        return False
                if(features[self.indx_split_feature[0]]>self.indx_split_feature[1]): # not the same
              #      print("\nwe are here h = 1\n")
                #    print("\n whatever left_node.h= : ",self.left_node.h)
                    n_samples = len(self.left_node.labels)
                    n_c1 = self.left_node.labels.sum()
                    n_c0 = n_samples-n_c1
                    if(n_c1>n_c0 ):
                        return True
                    else :
                        return False
            
            
            
            
                
                
            
        else :
    
                
            if (self.points.types[self.indx_split_feature[0]] == FeaturesTypes.BOOLEAN):     ## boolean type
                if(features[self.indx_split_feature[0]]==1):
                    return self.right_node.decide(features)
                if(features[self.indx_split_feature[0]]==0):
                    return self.left_node.decide(features)
              
            
                   # print("\nwe are not here yet  h =", self.h,"\n")
            if(self.points.types[self.indx_split_feature[0]] == FeaturesTypes.CLASSES) : ## classes type
                if(features[self.indx_split_feature[0]]==self.indx_split_feature[1]):
                    return self.right_node.decide(features)
                if(features[self.indx_split_feature[0]]!=self.indx_split_feature[1]):
                    return self.left_node.decide(features)                
                
            
            if(self.points.types[self.indx_split_feature[0]] == FeaturesTypes.REAL) : ## classes type
                if(features[self.indx_split_feature[0]]<=self.indx_split_feature[1]):
                    return self.right_node.decide(features)
                if(features[self.indx_split_feature[0]]>self.indx_split_feature[1]):
                    return self.left_node.decide(features)                
                 
            
       
                
                
            #print("\n didn't return :'(  h =", self.h,"\n")
        return predicted_label
        # raise NotImplementedError('Implement this method for Question 4')

