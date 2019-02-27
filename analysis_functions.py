''' 
PCA Plotting
Author: Tristan Miller
This contains code collected from my many notebooks
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

#performs all of the functions necessary to calculate and load the impact factor
def get_impact():
    num_games, game_gains, total_gains, card_list, card_dict = init_data()
    card_weights = get_card_weights(card_list,card_dict)
    card_impact = calculate_impact(game_gains,num_games,card_weights,card_dict)
    return sort_metric( card_impact, card_list, card_dict )

def get_synergy():
    num_games, game_gains, total_gains, card_list, card_dict = init_data()
    #card_weights = get_card_weights(card_list,card_dict)
    card_synergy = calculate_synergy(game_gains,num_games,card_list,card_dict)
    return sort_metric( card_synergy, card_list, card_dict )

#load pickles with data parsed from game logs
#num_games, game_gains, total_gains, card_list, card_dict = init_data()
def init_data():
    with open('gain_matrices.pkl','rb') as f:
        num_games, game_gains, total_gains = pickle.load(f)
    with open('card_list.pkl','rb') as f:
        card_list,card_dict = pickle.load(f)
    return num_games, game_gains, total_gains, card_list, card_dict

#calculates card weights based on a priori analysis of rules
#see impact factor notebook for explanation
#card_weights = get_card_weights(card_list,card_dict)
def get_card_weights(card_list,card_dict):
    card_weights = np.zeros((len(card_list),len(card_list)))
    #For any given kingdom card, the probabilities are roughly independent
    card_weights += 10/206
    #Some cards are in every kingdom
    card_weights[:,card_dict['Copper']] = 1
    card_weights[:,card_dict['Silver']] = 1
    card_weights[:,card_dict['Gold']] = 1
    card_weights[:,card_dict['Estate']] = 1
    card_weights[:,card_dict['Duchy']] = 1
    card_weights[:,card_dict['Province']] = 1
    card_weights[:,card_dict['Curse']] = 1
    #Colony and Platinum are 2.5 times as common (and I won't worry about the correlation with Prosperity cards)
    card_weights[:,card_dict['Colony']] *= 2.5
    card_weights[:,card_dict['Platinum']] *= 2.5
    card_weights[card_dict['Colony'],card_dict['Platinum']] = 1
    card_weights[card_dict['Platinum'],card_dict['Colony']] = 1
    #Ruins are three times as common, and guaranteed when a looter is in the kingdom
    card_weights[:,card_dict['Ruins']] *= 3
    card_weights[card_dict['Death Cart'],card_dict['Ruins']] = 1
    card_weights[card_dict['Marauder'],card_dict['Ruins']] = 1
    card_weights[card_dict['Cultist'],card_dict['Ruins']] = 1
    card_weights[card_dict['Ruins'],card_dict['Death Cart']] = 1/3
    card_weights[card_dict['Ruins'],card_dict['Marauder']] = 1/3
    card_weights[card_dict['Ruins'],card_dict['Cultist']] = 1/3
    #Potions are 9 times as common, and guaranteed if one of the potion cards is in the kingdom
    card_weights[:,card_dict['Potion']] *= 9
    potion_cards = ['Transmute','Scrying Pool','University','Apothecary','Familiar','Alchemist',"Philosopher's Stone",'Golem','Possession']
    for card in potion_cards:
        card_weights[card_dict[card],card_dict['Potion']] = 1
        card_weights[card_dict['Potion'],card_dict[card]] = 1/9
    #Finally, every card is guaranteed to appear in a game with itself
    np.fill_diagonal(card_weights,1)
    return card_weights
    
#calculate the impact factor
#can also replace argument game_gains with total_gains, if you want to weight it by how often the card was gained
def calculate_impact(game_gains,num_games,card_weights,card_dict):
    #element-wise division
    prc_gains = game_gains / num_games

    #Apply weighting. Multiply by card_weights[col], except along the diagonal of the matrix
    impact_prc = prc_gains * card_weights

    #Now from each row, subtract the vector from the average game
    copper_prc = impact_prc[card_dict['Copper'],:].copy()
    impact_prc -= copper_prc

    #Finally, calculate the impact factor for each card
    card_impact_prc = np.sum(abs(impact_prc),axis = 1)
    return card_impact_prc

def calculate_synergy(game_gains,num_games,card_list,card_dict):
    card_excluder = np.ones((len(card_list)))
    non_kingdom_cards = ['Copper','Silver','Gold','Estate','Duchy','Province','Curse','Colony','Platinum','Ruins','Potion','Prince','Walled Village']
    for card in non_kingdom_cards:
        card_excluder[card_dict[card]] = 0
        
    #element-wise division
    prc_gains = game_gains / num_games

    #Exclude non-kingdom cards by using card_excluder
    synergies = prc_gains * card_excluder

    #Now from each row, subtract the vector from the average game
    base_gain = synergies[card_dict['Copper'],:].copy()
    synergies -= base_gain

    #Finally, calculate the synergy factor for each card
    syn_factor = np.sum(abs(synergies),axis = 1)
    return syn_factor
    
#sorts any given metric and returns a pandas table
def sort_metric( metric, card_list, card_dict ):
    cards_sorted = sorted(card_list , key = lambda card: -metric[card_dict[card]])
    sorted_metric = sorted(metric, reverse=True)
    table = pd.DataFrame({'Rank':range(1,len(sorted_metric)+1),'Card':cards_sorted,'metric':sorted_metric})
    table = table.set_index('Rank')
    return table

#creates a dataframe that has both the qvist rankings, and the rankings based on a metric of choice
def compare_to_qvist(metric,qvist,category,year):
    #take the relevant subset of qvist rankings
    qvist_compare = qvist[np.logical_and(qvist['Category'] == category,np.logical_not(np.isnan(qvist[year])))]
    qvist_compare.reset_index(inplace=True,drop=True)

    #we only care about the cards that overlap with our impact data set
    overlap = [card in metric['Card'].tolist() for card in qvist_compare['Card']]

    #the overlap is 100% for 2014, but might have some false values in other cases
    qvist_compare = qvist_compare[overlap]
    qvist_compare.reset_index(inplace=True,drop=True)

    #next we take the relevant subset of the metric-based rankings
    overlap = [card in qvist_compare['Card'].tolist() for card in metric['Card']]
    metric_compare = metric[overlap]
    metric_compare.reset_index(inplace=True,drop=True)

    #now join this with qvist rankings
    qvist_compare['Metric ranking'] = np.nan
    for i,card in zip(qvist_compare.index,qvist_compare['Card']):
        qvist_compare.loc[i,'Metric ranking'] = np.where(metric_compare['Card'] == card)[0][0]+1
    return qvist_compare

#returns a score for the given metric
def score_metric(metric,qvist,year):
    categories = ['0-2 cost','3 cost','4 cost','5 cost','6+ cost','Potion cost']
    weight = np.zeros(6)
    corr = np.zeros(6)
    for i,category in enumerate(categories):
        qvist_compare = compare_to_qvist(metric,qvist,category,year)
        weight[i] = qvist_compare.shape[0]
        corr[i] = np.corrcoef(qvist_compare['Metric ranking'], qvist_compare[year])[0,1]
    
    corr *= weight
    corr /= sum(weight)
    return sum(corr)

#corrects the one spelling anomaly with jack of all trades
def correct_rankings(ranking):
    joat_index = np.where(ranking['Card'] == 'JackOfAllTrades')[0][0]
    ranking.loc[joat_index,'Card'] = 'Jack of all Trades'
    return ranking