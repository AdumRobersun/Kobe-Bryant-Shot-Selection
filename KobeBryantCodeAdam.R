#########################################################################
#########################################################################
#-------------KOBE BRYANT SHOT SELECTION KAGGLE COMPETITION-------------#
#########################################################################
#########################################################################


#Load in all relevant libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(embed)
library(forecast)
library(discrim)
library(naivebayes)
library(kknn)



#Import data and sample submission format
KB_data <- vroom('STAT348/Kobe-Bryant-Shot-Selection/data.csv')


#-----Feature Engineering-----#

#shotangle column
loc_x_zero <- KB_data$loc_x == 0
KB_data['angle'] <- rep(0,nrow(KB_data))
KB_data$angle[!loc_x_zero] <- atan(KB_data$loc_y[!loc_x_zero] / KB_data$loc_x[!loc_x_zero])
KB_data$angle[loc_x_zero] <- pi / 2

#Second remaining in game
KB_data$time_remaining = (KB_data$minutes_remaining*60)+KB_data$seconds_remaining

#distance from hoop
dist <- sqrt((KB_data$loc_x/10)^2 + (KB_data$loc_y/10)^2)
KB_data$shot_distance <- dist

# Home and Away
KB_data$matchup = ifelse(str_detect(KB_data$matchup, 'vs.'), 'Home', 'Away')

# Season
KB_data['season'] <- substr(str_split_fixed(KB_data$season, '-',2)[,2],2,2)

# Game number
KB_data$game_num <- as.numeric(KB_data$game_date)

# Achilles injury before and after
KB_data$postachilles <- ifelse(KB_data$game_num > 1452, 1, 0)

# MVP
KB_data$mvp <- ifelse(KB_data$game_num >= 909 & KB_data$game_num <= 990, 1, 0)

### period into a factor
KB_data$period <- as.factor(KB_data$period)

# delete columns
KB_data_feat <- KB_data %>%
  select(-c('shot_id', 'team_id', 'team_name', 'shot_zone_range', 'lon', 'lat',
            'seconds_remaining', 'minutes_remaining', 'game_event_id',
            'game_id', 'game_date','shot_zone_area',
            'shot_zone_basic', 'loc_x', 'loc_y'))

# KB_train
KB_train <- KB_data_feat %>%
  filter(!is.na(shot_made_flag))

# KB_test
KB_test <- KB_data_feat %>%
  filter(is.na(shot_made_flag))

# Make the response variable into a factor
KB_train$shot_made_flag <- as.factor(KB_train$shot_made_flag)

recipe <- recipe(shot_made_flag ~ ., data = KB_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors())

KB_test.id <- KB_data %>% filter(is.na(shot_made_flag)) %>% select(shot_id)





###################################
#----MODEL ONE: RANDOM FOREST-----#
###################################



## Create a workflow with model & recipe

KB_rf_mod <- rand_forest(mtry = tune(),
                         min_n=tune(),
                         trees=100) %>%
  set_engine("ranger") %>%
  set_mode("classification")


KB_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(KB_rf_mod)

#Set up grid of tuning values

tuning_grid <- grid_regular(mtry(range = c(1,(ncol(KB_train)-1))),
                            min_n(),
                            levels = 3)
#Set up K-fold CV

folds <- vfold_cv(KB_train, v = 3, repeats=1)

#Find best tuning parameters

CV_results <- KB_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

#selection metric for tuning
best_tune_rf <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and predict

final_rf_wf <- KB_wf %>%
  finalize_workflow(best_tune_rf) %>%
  fit(data=KB_train)

KB_rf_predictions <- final_rf_wf %>% predict(new_data=KB_test,
                                            type="prob")

KB_randomforest_preds <- as.data.frame(cbind(KB_test.id, as.character(KB_rf_predictions$.pred_1)))

colnames(KB_randomforest_preds) <- c("shot_id", "shot_made_flag")

write_csv(KB_randomforest_preds, "KB_randomforest_preds.csv")







#########################################
#----MODEL TWO: K-NEAREST NEIGHBORS-----#
#########################################

#Kobe K nearest neighbors workflow

KB_knn_mod <- nearest_neighbor(neighbors=tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

KB_knn_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(KB_knn_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(neighbors(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(KB_train, v = 5, repeats=1)

## Run the CV
CV_results <- KB_knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL


# Find Best Tuning Parameters
best_tune_knn <- CV_results %>%
  select_best("roc_auc")

#Finalize the wf and fit
final_knn_wf <- KB_knn_wf %>%
  finalize_workflow(best_tune_knn) %>%
  fit(data=KB_train)

#Find Predictions for KNN

KB_data_predictions_knn <- final_wf %>% predict(new_data=KB_test,
                                             type="prob")

KB_knn_preds <- as.data.frame(cbind(KB_test.id, as.character(KB_data_predictions_knn$.pred_1)))

colnames(KB_knn_preds) <- c("shot_id", "shot_made_flag")

write_csv(KB_knn_preds, "KB_knn_preds.csv")



################################################
#-----MODEL THREE: SUPPORT VECTOR MACHINES-----#
################################################


KB_svm_mod <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kernlab")

KB_svm_wf <- workflow() %>% 
  add_recipe(recipe) %>% 
  add_model(KB_svm_mod)

svm_tuning_grid <- grid_regular(rbf_sigma(), cost(),levels = 5)

folds <- vfold_cv(KB_train, v = 5, repeats=1)

KB_svm_results <- KB_svm_wf %>% 
  tune_grid(resamples = folds,
            grid = svm_tuning_grid,
            metrics = metric_set(roc_auc))

svm_bestTune <- KB_svm_results %>% 
  select_best("roc_auc")

KB_svm_final_wf <- KB_svm_wf %>% 
  finalize_workflow(svm_bestTune) %>% 
  fit(data=KB_train)

KB_svm_preds <- predict(KB_svm_final_wf,
                    new_data=KB_test,
                    type="prob")

KB_SVM_Preds <- as.data.frame(cbind(KB_test$id, KB_svm_preds$.pred_1))
colnames(KB_SVM_Preds) <- c("shot_id", "shot_made_flag")
write_csv(KB_SVM_Preds, "KB_SVM_Preds.csv")



########################################################################
########################################################################
