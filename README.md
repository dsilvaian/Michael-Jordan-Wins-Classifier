# Michael-Jordan-Wins-Classifier
To predict whether a game that Michael Jordan played in resulted in a win (or loss) for his team. We determined this by building classification models using Jordan’s statistical data for each game.

# Dataset Description
This dataset contains game statistics for every regular season NBA game that Michael Jordan played in his
career. One of these statistics, “Win”, is whether or not the game resulted in a win for Jordan’s team.

EndYear: Ending year of the season the game was played (e.g. 1995-1996season is 1996)
G: Number game Jordan played in the season (e.g. the 3rd game Jordan played in the 1997-1998 season is 3)
Date: Date the game was played
Years: Years old Jordan was for the game
Days: Days past the year Jordan was for the game
Age: Exact age Jordan was when the game was played
Tm: Team Jordan was playing for
Home: Whether or not the game was played at Jordan<92>s team’s Homearena
Opp: Opponent (i.e. Opposing team) for the game
Win: Whether or not Jordan<92>s team won
Diff: Score differential for the game (positive numbers are wins, negative numbers are losses)
GS: Whether or not Jordan was in the starting lineup for the game
MP: Minutes Jordan played in the game
FG: Total field goals made by Jordan in the game
FGA: Total field goals attempted by Jordan in the game
FG_PCT: Percentage of field goals made by Jordan in the game
3P: 3-Point field goals made by Jordan in the game
3PA: 3-Point field goals attempted by Jordan in the game
3P_PCT: Percentage of 3-Point field goals made by Jordan in the game
FT: Free throws made by Jordan in the game
FTA: Free throws attempted by Jordan in the game
FT_PCT: Percentage of free throws made by Jordan in the game
ORB: Offensive rebounds grabbed by Jordan in the game
DRB: Defensive rebounds grabbed by Jordan in the game
TRB: Total rebounds grabbed by Jordan in the game
AST: Assists distributed by Jordan in the game
STL: Steals made by Jordan in the game
BLK: Blocks made by Jordan in the game
TOV: Turnovers caused by Jordan in the game
PF: Personal fouls committed by Jordan in the game
PTS: Points scored by Jordan in the game
GmSc: Game Score; the formula is PTS + 0.4 * FG - 0.7 * FGA - 0.4*(FTA - FT) + 0.7 * ORB + 0.3 * DRB + STL + 0.7 * AST + 0.7 * BLK - 0.4 * PF - TOV. Game Score was created by John Hollinger to give a rough measure of a player’s productivity for a single game. The scale is similar to that of points scored, (40 is an outstanding performance,10 is an average performance, etc.)

# Dataset Link: https://sports-statistics.com/sports-data/sports-data-sets-for-data-modeling-visualization-predictions-machine-learning/
