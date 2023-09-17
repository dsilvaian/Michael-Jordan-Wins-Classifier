## Final Project
## Ian D'Silva & David Vanderschaaf
## 4/5/2023

##Ref:https://sports-statistics.com/sports-data/sports-data-sets-for-data-modeling-visualization-predictions-machine-learning/
library(dplyr)
library(rsample)
setwd("D:/Boston/CS 699/Project")
#setwd("../Final Project")


#Import data-set
mjStats <- read.csv("michael-jordan-nba-career-regular-season-stats-by-game_Original.csv")

## Pre-processing Data
# Remove dependent and duplicate attributes
dupAttributes <- c("Rk", "Years", "Days", "Diff", "FG", "X3P", "FT", "TRB")
mjStats <- select(mjStats, -dupAttributes)

# Check for missing values
any(missing(mjStats))
any(is.na(mjStats))

# After inspection, update the NA's to 0.0 since NA values are due to divide by
# zero error
mjStats[is.na(mjStats)] <- 0.0

# Setting seed to produce consistent split for data-set
seed <- 100
set.seed(seed)

# Splitting the data-set into train and test
dataset_split <- initial_split(mjStats, prop = 0.66, strata = mjStats$Win)
train_mjStats <- training(dataset_split)
test_mjStats <- testing(dataset_split)
