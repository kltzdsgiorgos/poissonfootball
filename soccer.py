import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson, skellam
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Chooses the location of the
file_list = os.listdir(r".")
print(file_list)

epl2018 = pd.read_csv(input("Choose league:"))  # choose the .csv file
epl2018 = epl2018[['Home', 'Away', 'FTHG', 'FTAG', ]]  # choose the colum names as writen in the .csv file
epl2018 = epl2018.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})  # rename the columns

goal_model_data = pd.concat([epl2018[['Home', 'Away', 'HomeGoals']].assign(home=1).rename(
    columns={'Home': 'team', 'Away':'opponent', 'HomeGoals': 'goals'}),
    epl2018[['Away', 'Home', 'AwayGoals']].assign(home=0).rename(
        columns={'Away': 'team', 'Home': 'opponent', 'AwayGoals': 'goals'})])

poisson_model = smf.glm(formula = "goals ~ home + team + opponent", data=goal_model_data,
                        family=sm.families.Poisson()).fit()

# prints a summary of the data
print(poisson_model.summary())


def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team':homeTeam, 'opponent': awayTeam,'home':1},index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team':awayTeam, 'opponent': homeTeam, 'home':0},index=[1])).values[0]

    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in[home_goals_avg, away_goals_avg]]
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))


homelist = []

print("Enter the home team names, when finished type 'end'")

donehome = False

while not donehome:
    homeT = input()
    homelist.append(homeT)
    if homeT == 'end':
        del homelist[-1]
        donehome = True

awaylist = []

print("Enter the away team names, when finished type 'end'")

doneaway = False

while not doneaway:
    awayT = input()
    awaylist.append(awayT)
    if awayT == 'end':
        del awaylist[-1]
        doneaway = True
# print(', '.join(homelist), "AGAINST ",end="")
# print(', ',join(awaylist))
# homeT = input("Choose Home Team:")
# awayT = input("Choose Away Team:")
for homeT, awayT in zip(homelist, awaylist):
    # print(simulate_match(poisson_model, homeT, awayT, max_goals=5)) # Prints the simulation array of goals
    match = simulate_match(poisson_model, homeT, awayT, max_goals = 10)
    print("#################################")
    print("%s Win:" % homeT, end="")
    print(" Odds",round(1/(np.sum(np.tril(match, -1))),3), "OR",round((np.sum((np.tril(match, -1)))*100),3),"%")
    print("Draw:", end="")
    print(" Odds",round(1/(np.sum(np.diag(match))),3), "OR", round((np.sum(np.diag(match))*100),3),"%")

    print("%s Win:" % awayT, end="")
    print(" Odds",round(1/(np.sum(np.triu(match, 1))),3), "OR", round((np.sum(np.triu(match, 1)))*100,3),"%")

    print("Over 2.5 goals:", end="")
    print(" Odds",round(1/(np.sum(match[2:])+np.sum(match[:2,2:])-np.sum(match[2:3,0])-np.sum(match[0:1,2])),3), "OR", round((np.sum(match[2:])+np.sum(match[:2,2:])-np.sum(match[2:3,0])-np.sum(match[0:1,2])),3),"%")

    print("Under 2.5 goals:", end="")
    print(" Odds",round((1/(np.sum(match[:2,:2])+match.item((0,2))+match.item((2,0)))),3), "OR",round((np.sum(match[:2,:2])+match.item((0,2))+match.item((2,0))),3),"%")

    print("Home Clean Sheet Yes:", end="")
    print(" Odds",round(1/(np.sum(match[:,0])),3), "OR",round((np.sum(match[:,0])),3),"%")

    print("Home Clean Sheet No:", end="")
    print(" Odds", round(1/(np.sum(match[:,1:])),3), "OR",round((np.sum(match[:,1:])),3),"%")

    print("Away Clean Sheet Yes:", end="")
    print(" Odds", round(1/(np.sum(match[:1])),3), "OR",round((np.sum(match[:1])),3),"%")

    print("Away Clean Sheet No:", end="")
    print(" Odds",round(1/(np.sum(match[1:])),3), "OR",round((np.sum(match[1:])),3),"%")

    print("Both Teams score Yes:", end="")
    print(" Odds",round(1/(np.sum(match[1:,1:])),3), "OR",round((np.sum(match[1:,1:])),3),"%")

    print("Both Teams score No:", end="")
    print(" Odds",round(1/(np.sum(match[:1])+np.sum(match[1:,0])),3), "OR",round(((np.sum(match[:1])+np.sum(match[1:,0]))),3),"%")
    print("#################################")
