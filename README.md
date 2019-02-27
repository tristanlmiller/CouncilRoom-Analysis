# CouncilRoom Analysis

## Introduction

This is a project by Tristan Miller ([@tristanlmiller](https://github.com/tristanlmiller/)) from May 2018.  The goal was to apply machine learning and other analysis techniques to game logs of *Dominion*, in order to characterize cards by their strength and type.

*Dominion* is a [popular card game](https://boardgamegeek.com/boardgame/36218/dominion) created in 2008.  In each game of *Dominion*, there is a unique set of cards available for players to buy, and the utility of each card depends on the environment.  Over the course of many games, expert players get a sense of which cards are "stronger" than others.  The community's subjective views of card strengths are measured each year by the [Qvist Rankings](http://wiki.dominionstrategy.com/index.php/List_of_Cards_by_Qvist_Rankings).

The problem with the community rankings is that subjective opinions can be influenced by selective recall and personal bias.  But on the other hand, it is also challenging to come up with any objective metric for card strength, because it's not clear what that means.  My goal is to come up with a new metric, and test how well it correlates with the community rankings.

I analyzed ~140,000 game logs taken from CouncilRoom.com, in order to define the **impact factor** of each card.  These game logs were restricted to games with top-ranking players, and the impact factor is a measure of how much the presence of a card changes player behavior.  The impact factor is a closer match to the Qvist Rankings than any other objective measure we know of.  This provides a lot of insight into what it intuitively means for a card to be "strong".

Additionally, using the insight gained in measuring the impact factor, I define a list of features for each card.  By applying unsupervised machine learning, I characterized cards by their type.

My analysis of the impact factor inspired other data-minded people in the *Dominion* community, who have collected statistics from more recent *Dominion* game logs.  The new analysis [is publicly available](https://docs.google.com/spreadsheets/d/1M2L7hcY3sbA33OwuZhgPYJWVlMFgJYBdK8cnkbJHmbo/edit#gid=1032809554), and includes a calculation of the impact factor for each card.

## Project Outline

1. I parsed ~140k game logs, in order to extract the set of cards gained by each player in each game. 
2. I calculated the impact factor for each card.  I ranked the cards by impact factor and shared the list [on the *Dominion* forums](http://forum.dominionstrategy.com/index.php?topic=18577.0) for feedback.
3. I defined synergy relations between pairs of cards, and used this to create a list of features for each card.  The features are weighted to distinguish between types of cards, as opposed to power levels of cards.  I applied PCA and semi-supervised clustering to these features.
4. I generated graphs showing the PCA and cluster analysis.  These were shared [on the *Dominion* forums](http://forum.dominionstrategy.com/index.php?topic=18629.0) for feedback.

## CouncilRoom data

[CouncilRoom.com](http://councilroom.com/) has a repository of game logs taken from Isotropic, an online implementation of Dominion that ran from 2010-2013.  This data includes all expansions up to *Dominion: Guilds*, and not later expansions.  All in all, this includes 206 distinct kingdom piles, and 11 additional supply piles (e.g. the base cards, Potion, Ruins).

CouncilRoom.com hosts some of their own analyses of the game logs, but there have also been several analyses conducted by other people:

1. [How often do top players gain each card?](http://forum.dominionstrategy.com/index.php?topic=12341.0)
2. [What are the win rates conditional on gaining each card?](http://forum.dominionstrategy.com/index.php?topic=12475.0)
3. [How much does player skill correlate with gaining each card?](http://forum.dominionstrategy.com/index.php?topic=12351.0)
4. [How hard it is to predict the winner from player rankings, in games that include each card?](http://forum.dominionstrategy.com/index.php?topic=2798.msg47781#msg47781)

My data set consists of ~140,000 game logs from CouncilRoom.com.  All games are 2 players, pro rated, no bots, and each game has at least one player ranked in the top 100 by Isotropish.  I was provided with this data by forum-goer ben_king, and did not do the work of selecting the relevant logs.  The game logs are about 2.5 GB so they are excluded from this repository.

Each game log has a complete set of information about the game.  They list the players' names, the available cards, and a sequence of every single move made by each player.  I parsed each game log to extract only the specific information that I needed, while compiling this information into summary statistics.  See "Parsing game logs.ipynb" for code and details.

## Impact factor

Impact factor is formally defined by the following equation:

Impact(X) = sum_Y | P(gain(Y)|supply(X)) - P(gain(Y)) |

X and Y are supply piles, and gain(Y) is the subset of games where Y is gained at least once, and supply(X) is the subset of games where X is available to be gained.  Further details are shown in "Impact factor.ipynb".

I calculated the impact factor, and made card rankings shown [here](http://forum.dominionstrategy.com/index.php?topic=18577.0).  

The list more or less matches player intuitions about which cards are strong or weak.  The fact that the impact factor so closely reflects subjective judgments of card strength suggests that they may be conceptually similar.  People perceive a card to be strong when it distorts games.  The impact factor seems to be a useful way to quantify that distortion.

There are a few notable biases in the impact factor.  It appears to give more of an advantage towards cards that hand out bad cards.  For instance, Ill-Gotten Gains forces opponents to gain Curses, and makes it easy to gain Coppers.  Coppers and Curses are present in every game, but players wouldn't normally wouldn't gain them willingly.  Thus, Ill-Gotten Gains drastically changes the gain frequencies of Copper and Curse, which may disproportionately boost its impact factor.  A possible path for improvement is to apply weightings that de-emphasize cards like Copper or Curse.  However, I felt it was best to calculate the metric in the most intuitive and transparent way, without any free parameters.  

### Evaluation

To evaluate how well the impact factor correlates with card strength, I calculated the correlation coefficient between rankings based on the impact factor, and the 2014 Qvist rankings.  The Qvist rankings are broken up into several lists, so I just calculated the correlation coefficient for each one and took a weighted average.  These calculations can be found in "Metric evaluation.ipynb"

The impact factor performs slightly better than the most popular objective metric for card strength, the gain percentage of each card.  The average correlation coefficient for impact factor is 0.863, while that of the gain percentage rankings is 0.846.  This modest improvement holds true for the earlier years of Qvist rankings, but oddly it is a worse predictor for Qvist rankings 2015 onward.  Of course, since the data only extends out to 2013, you wouldn't expect any metric to be a good predictor of later rankings.

I suspect that the best way to model community rankings would be to calculate a weighted sum of various metrics.  However, if we did that, then we might worry about overfitting the rankings, subjective biases and all.  I have not performed this analysis yet, but there is a simple solution.  After fitting the metric to the 2014 rankings, the metric can then be evaluated on the 2013 and 2015 rankings.  If the correlation with 2014 improves, but the correlation with 2013 and 2015 get worse, this suggests overfitting.

## Principal Component Analysis

For details on this analysis, see "Card PCA.ipynb".  Scatter plots are shown in the "Images" folder, and more plots can be generated using the code in "Graphs.ipynb"

In order to calculate a list of features for use in PCA, I defined two synergy relations, the "promote" and "love" relations (see glossary below).  Since there are 206 kingdom piles, each kingdom pile has 412 features.  The first 206 features are how much that pile promotes each other pile, and the last 206 features are how much that pile loves each other pile.

One problem is that low-impact cards, by definition, barely promote or anti-promote any other cards.  Thus any analysis of these features would likely group all the low-impact cards together, regardless of type.  To resolve this problem, I normalized each feature vector, ensuring that the Euclidean norm of the "promote" vector is 1, and the norm of the "love" vector is also 1.  Finally, in order to improve statistics, I applied weighting to reduce the impact of cards for which I have less data.

Here are a couple plots of the first 4 components:

<div align="center"><img src="Images/c1 vs c2.png" alt="Components 1 and 2" width=900></div>
<div align="center"><img src="Images/c3 vs c4.png" alt="Components 2 and 3" width=900></div>

Note that in each graph, the different cards are assigned different colors.  These colors are based on Label Propagation, using cluster seeds that I chose.  The clusters are not entirely accurate reflections of commonly discussed card categories, but are simply there to make the graphs more interpretable by experienced *Dominion* players.

By inspection, I can determine the meaning of the first 6 components:

1. Important terminal cards vs villages
2. 5-cost cards, vs things that help you get 5-costs
3. Thinners vs cards that are best in thin decks
4. Cards that promote "slog" strategies vs cards that promote "engine" strategies
5. Trashing vs draw
6. Cheap cards and defense cards vs strong attacks and gainers

The fraction of variance explained by each component was [0.24399325  0.20182445  0.15005617  0.10892099  0.09377003  0.08707616].  Further components were unintelligible to me.

I can also show which cards are most loved or most promoted by each component.  For instance, we can show that cards with high component 1 tend to love and promote cards with low component 1.  What this means is that cards on opposite ends of component 1 work best when mixed together in the same deck.  On the other hand, the opposite is true of component 4.  Cards with high component 4 tend to love and promote each other.  This suggests that there are divergent strategies, with some decks mostly gaining cards that are high in component 4, and some decks mostly gaining cards that are low in component 4.

<div align="center"><img src="Images/c1 vs c1L.png" alt="Component 1 vs Loved by Component 1" width=900></div>
<div align="center"><img src="Images/c4 vs c4L.png" alt="Component 4 vs Loved by Component 4" width=900></div>

This analysis was shared and discussed [on the *Dominion* forums](http://forum.dominionstrategy.com/index.php?topic=18629.0).

## Conclusions

My analysis suggests that cards in *Dominion* are perceived to be strong when their presence distorts player behavior.  The PCA analysis suggests what factors are most important to consider when understanding the type of a particular card.

I will go on to suggest that this type of analysis could be useful in programming AI for *Dominion*.  Because of the sheer number of cards with very different effects, it is very difficult for an AI to predict synergies between cards, based solely on the cards' rules.  The simpler approach may be to compile statistics from human players, in order to highlight the most important synergies.

## Glossary

These are technical terms that I use throughout the repository.

**Kingdom piles** - These are piles of cards that are available to players, selected at random at the beginning of each game.  Most piles consist of multiple copies of the same card, but a few piles consist of distinct cards.  I use piles as the unit of analysis, rather than distinct cards.

**Supply piles** - A kingdom pile is a type of supply pile, but there are also special supply piles that either appear in every game, or which appear contingent on other supply piles. (e.g. Province, Potion, Ruins)

**Gain frequency** - The gain frequency of a card is the probability that, given a random game and one of the players, that player gains the card at least once.  The gain frequency of card X is notated P(gain(X)).

**Gain percentage** - The gain percentage of a card is the probability that, given a random game *where that card is available* and one of the players, that player gains the card at least once.  The gain percentage of card X is notated P(gain(X)|supply(X)).

**Impact factor** - The impact factor of a supply pile tells you how much the presence of that card changes gain frequencies, relative to the average gain frequency.  The formal definition is:

Impact(X) = sum_Y | P(gain(Y)|supply(X)) - P(gain(Y)) |

**Synergy factor** - The synergy factor is similar to the impact factor, but calculated from the gain percentages rather than gain frequencies.  The formal definition is:

Synergy(X) = sum_Y | P(gain(Y)|supply(X)&supply(Y)) - P(gain(Y)|supply(Y)) |

**Promotion relation** - For two kingdom piles X and Y, I say that X promotes Y if the gain percentage of Y is increased when X is present.  The formal definition is:

prom(X,Y) = P(gain(Y)|supply(X)&supply(Y)) - P(gain(Y)|supply(Y))

**Love relation** - For two kingdom piles X and Y, I say that X loves Y if the gain percentage of X is increased relative to other piles, when Y is present.  This is notated love(X,Y).  love(X,Y) is related to prom(Y,X) by the following expression:

love(X,Y) = prom(Y,X) - sum_Z (prom(Z,X)/N)

**Principal Component Analysis (PCA)** - PCA is a standard unsupervised machine learning technique.  It tries to describe the variance in a set of data using a small number of dimensions ("components").

**Cluster analysis** - Cluster Analysis is a type of machine learning technique that classifies objects based on how they cluster together.  There are multiple types of cluster analysis, but here we use Label Propagation.  This is a semi-supervised algorithm that allows the user to define seeds for each cluster, which are then grown to classify all objects.