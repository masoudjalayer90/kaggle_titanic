
# base packages
library(dplyr)
library(Amelia)
library(caTools)
library(pscl)
library(pROC)
library(ggplot2)

# decision trees 
library(rpart)
library(rpart.plot)

# bagging
library(ipred)
library(vip)
library(pdp)
library(caret)

# random forrests
library(h2o)
library(ranger)



getwd()
setwd('C:/Users/chemaja/Documents/Personal/kaggle/Titanic/data')

## import data into R 

test_raw <- read.csv('test.csv', na.strings = c(''))
train_raw <- read.csv('train.csv', na.strings = c(''))

glimpse(train_raw)
summary(train_raw)

# data clean
sapply(train_raw, function(x) sum(is.na(x)))
sapply(train_raw, function(x) length(unique(x)))

missmap(train_raw)

df <- subset(train_raw, select = c(2, 3, 5, 6, 7, 8, 10, 12))

df$Age[is.na(df$Age)] <- mean(df$Age, na.rm = TRUE)

df <- df[!is.na(df$Embarked),]
 
split <- sample.split(df, SplitRatio = 0.8)

df_train <- subset(df, split == TRUE)
df_test <- subset(df,split == FALSE)

# simple log regression model ----------------------------
 
model_glm <- glm(Survived ~ ., family = binomial(link = 'logit'), 
                 data = df_train)

summary(model_glm)

anova(model_glm, test = 'Chisq')

pR2(model_glm)

glimpse(df_test)

fitted_results_glm <- predict(model_glm, newdata = subset(df_test, select = c(2,3,4,5,6,7,8)), type = 'response')

fitted_results_glm <- ifelse(fitted_results_glm > 0.5, 1, 0)

miss_classified_glm <- mean(fitted_results != df_test$Survived)
paste(print((1-miss_classified_glm)*100),'%',' Accurate')

# ROC curve
plot.roc(df_test$Survived, fitted_results_glm, main = 'glm_model')

# decision tree ------------------------------------------

model_tree <-  rpart(Survived ~., 
                     data = df_train, 
                     control = rpart.control(cp = 0.00001, 
                                             minsplit = 10))

printcp(model_tree)
plotcp(model_tree)
model_tree$variable.importance

bestcp <- model_tree$cptable[which.min(model_tree$cptable[,'xerror']),'CP']

model_tree_pruned <-prune(model_tree, cp = bestcp)

rpart.plot(model_tree_pruned)

fitted_results_tree <- predict(model_tree_pruned, newdata = df_test)

# fitted_results_tree <- ifelse(fitted_results_tree > 0.5, 1, 0)
miss_classified_tree <- mean(fitted_results_tree != df_test$Survived)
paste(print((1-miss_classified_tree)*100),'%',' Accurate')

# ROC curve
plot.roc(df_test$Survived, fitted_results_tree, main = 'tree_model')

# bagging ------------------------------------------------

# using ipred
model_bagging <-bagging(Survived ~ ., 
                        data = df_train, 
                        nbagg = 200, 
                        coob = T, 
                        control = rpart.control(minsplit = 5, 
                                                cp = 0))

# using caret
model_bagging2 <- caret::train(
  Survived ~ .,
  data = df_train,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 10),
  nbagg = 500,  
  control = rpart.control(minsplit = 2, cp = 0)
)


fitted_results_bagging2 <- predict(model_bagging2, newdata = df_test)

fitted_results_bagging2 <- ifelse(fitted_results_bagging2 > 0.5, 1, 0)
miss_classified_bagging2 <- mean(fitted_results_bagging2 != df_test$Survived)
paste(print((1-miss_classified_bagging2)*100),'%',' Accurate')

# ROC curve
plot.roc(df_test$Survived, fitted_results_bagging2, main = ' bagging_model')

# variable importance plot
vip::vip(model_bagging2, num_features = 40 , bar = FALSE)


# partial dependence plots
p1 <- 
  pdp::partial(
  model_bagging2, 
  pred.var = "Fare",
  grid.resolution = 20) %>% 
  autoplot()

p2 <- 
  pdp::partial(
    model_bagging2, 
    pred.var = "Age",
    grid.resolution = 20) %>% 
  autoplot()

p3 <- 
  pdp::partial(
    model_bagging2, 
    pred.var = "SibSp",
    grid.resolution = 20) %>% 
  autoplot()

p4 <- 
  pdp::partial(
    model_bagging2, 
    pred.var = "Parch",
    grid.resolution = 20) %>% 
  autoplot()


cowplot::plot_grid(p1,p2, p3, p4)


# ranodm forrests --------------------------------------------

# define mtry 
n_features <- length(setdiff(names(df_train), 'Survived'))

model_rforrest1 <- ranger(Survived ~ . , 
                          data = df_train,
                          respect.unordered.factors = 'order', 
                          classification = TRUE,
                          mtry = n_features **0.5)

default_rmse <- sqrt(model_rforrest1$prediction.error)


fitted_results_rforrest1 <- predict(model_rforrest1, data = df_test)

miss_classified_rforrest1 <- mean(fitted_results_rforrest1$predictions != df_test$Survived)
paste(print((1-miss_classified_rforrest1)*100),'%',' Accurate')

# ROC curve
plot.roc(df_test$Survived, fitted_results_rforrest1$predictions, main = 'random_forrest_model')


# create hyperparameter grid with the parameters to be tuned
 hyper_grid <- 
   expand.grid(mtry = floor(n_features * c( 0.15, 0.25, 0.33, 0.4, 0.5)), 
               min_node_size = c(1,3 , 5, 10), 
               replace = c(TRUE, FALSE), 
               sample.fraction = c(0.5, 0.63, 0.8, 0.85, 0.9, 0.95), 
               rmse = NA
               )


# execute search for different combinations of parameters

for(i in seq_len(nrow(hyper_grid))){
fit <- 
    ranger(Survived ~ ., 
           data = df_train, 
           num.trees = n_features * 10, 
           mtry = hyper_grid$mtry[i], 
           min.node.size = hyper_grid$min_node_size[i], 
           replace = hyper_grid$replace[i], 
           sample.fraction = hyper_grid$sample.fraction[i], 
           verbose = FALSE, 
           respect.unordered.factors = 'order', 
           seed = 123
           )
  hyper_grid$rmse[i] <- sqrt(fit$prediction.error)
    }

hyper_grid %>% 
  arrange(rmse) %>% 
  mutate(perc_gain = (default_rmse - rmse)/ default_rmse * 100) %>% 
  head(10)


model_rforrest2 <- ranger(Survived ~ . , 
                          data = df_train,
                          min.node.size = 3, 
                          replace = F, 
                          sample.fraction = 0.63,
                          respect.unordered.factors = 'order', 
                          classification = TRUE,
                          mtry = 2)

fitted_results_rforrest2  <- predict(model_rforrest2, data = df_test)

miss_classified_rforrest2 <- mean(fitted_results_rforrest2$predictions != df_test$Survived)
paste(print((1-miss_classified_rforrest2)*100),'%',' Accurate')

# ROC curve
plot.roc(df_test$Survived, fitted_results_rforrest2$predictions, main = 'random_forrest_tweaked_model')


