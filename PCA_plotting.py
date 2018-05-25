''' 
PCA Plotting
Author: Tristan Miller
This contains code to streamline the creation of PCA scatter plots
'''
#import os
#import re
#import pdb
import numpy as np
import pickle
#import time
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn.semi_supervised as semi
from sklearn.cluster import SpectralClustering as spectral
import adjustText

#When you create an instance of this class, it loads all the information about the PCA and clustering models
#this is used to create maps
class card_model:
    def __init__(self):
        with open('model_results.pkl','rb') as f:
            self.reduced_vectors, self.model_components, self.kingdom_list, self.cluster_labels = pickle.load(f)
            self.label_size = 10
            self.figure_size = (10,8)
            self.adjust_text = False
            self.repel_strength = 0.01
    
    def set_figure_size(self,width,height):
        self.figure_size = (width,height)
        
    def set_label_size(self,size):
        self.label_size = size
    
    #This can be set to True in order to adjust the positions of text labels to make them overlap less
    #this takes a long time, and is not always effective
    def set_adjust_text(self,adjust_text, repel_strength=0.01):
        self.adjust_text = adjust_text
        self.repel_strength = repel_strength
    
    def pca_map_color(self,axis1,axis2,axis1_type='card',axis2_type='card'):
        plt.figure(figsize=self.figure_size)
        if( axis1_type == 'card' ):
            component1 = self.reduced_vectors[:,axis1]
            plt.xlabel('Component %i' % (axis1+1),fontsize=16)
        elif( axis1_type == 'prom' ):
            component1 = self.model_components[axis1,0:len(self.kingdom_list)].transpose()
            plt.xlabel('Promoted by component %i' % (axis1+1),fontsize=16)
        elif( axis1_type == 'love' ):
            component1 = self.model_components[axis1,len(self.kingdom_list):len(self.kingdom_list)*2].transpose()
            plt.xlabel('Loved by component %i' % (axis1+1),fontsize=16)
        else:
            raise Exception('Must choose "card","prom", or "love" as axis type.')

        if( axis2_type == 'card' ):
            component2 = self.reduced_vectors[:,axis2]
            plt.ylabel('Component %i' % (axis2+1),fontsize=16)
        elif( axis2_type == 'prom' ):
            component2 = self.model_components[axis2,0:len(self.kingdom_list)].transpose()
            plt.ylabel('Promoted by component %i' % (axis2+1),fontsize=16)
        elif( axis2_type == 'love' ):
            component2 = self.model_components[axis2,len(self.kingdom_list):len(self.kingdom_list)*2].transpose()
            plt.ylabel('Loved by component %i' % (axis2+1),fontsize=16)
        else:
            raise Exception('Must choose "card","prom", or "love" as axis type.')

        plt.scatter(component1, component2,lw=0,s=30,c=self.cluster_labels,cmap=plt.cm.Dark2)
        
        texts = []
        for label, x, y in zip(self.kingdom_list, component1, component2):
            if(self.adjust_text):
                texts.append(plt.text(x,y,label,fontsize=self.label_size))
            else:
                plt.annotate(label, xy = (x,y),xytext = (0,self.label_size*0.6),
                             textcoords='offset points',ha='center',fontsize=self.label_size)
        if(self.adjust_text):
            adjustText.adjust_text(texts,force_text=self.repel_strength,arrowprops=dict(arrowstyle="-", color='k', lw=0.5),text_from_points=False,expand_text=(1,1))
        plt.show()