#Predicting credit default risk using Home Credit Group data provided by kaggle - https://www.kaggle.com/c/home-credit-default-risk/data

library(tidyverse)
library(dummies)
library(mice)
library(randomForest)
library(e1071)
library(class)
library(caret)

train_path <- "~/School/MSDS692 - Data Science Practicum/Home Credit Group/application_train.csv"
credit_train <- read_csv(train_path)
#note: data is unable to be provided here, visit link above to download the data

#explore the data
glimpse(credit_train)

#Difficult to see summary of character fields so they will be converted into factors before viewing the summary
#create factors
factor_list <- c("TARGET","NAME_CONTRACT_TYPE","CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY","NAME_TYPE_SUITE","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","FLAG_MOBIL","FLAG_EMP_PHONE","FLAG_WORK_PHONE","FLAG_CONT_MOBILE","FLAG_PHONE","FLAG_EMAIL","OCCUPATION_TYPE","REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY","WEEKDAY_APPR_PROCESS_START","HOUR_APPR_PROCESS_START","REG_REGION_NOT_LIVE_REGION","REG_REGION_NOT_WORK_REGION","LIVE_REGION_NOT_WORK_REGION","REG_CITY_NOT_LIVE_CITY","REG_CITY_NOT_WORK_CITY","LIVE_CITY_NOT_WORK_CITY","ORGANIZATION_TYPE","FLAG_DOCUMENT_2","FLAG_DOCUMENT_3","FLAG_DOCUMENT_4","FLAG_DOCUMENT_5","FLAG_DOCUMENT_6","FLAG_DOCUMENT_7","FLAG_DOCUMENT_8","FLAG_DOCUMENT_9","FLAG_DOCUMENT_10","FLAG_DOCUMENT_11","FLAG_DOCUMENT_12","FLAG_DOCUMENT_13","FLAG_DOCUMENT_14","FLAG_DOCUMENT_15","FLAG_DOCUMENT_16","FLAG_DOCUMENT_17","FLAG_DOCUMENT_18","FLAG_DOCUMENT_19","FLAG_DOCUMENT_20","FLAG_DOCUMENT_21","WALLSMATERIAL_MODE","EMERGENCYSTATE_MODE","FONDKAPREMONT_MODE","HOUSETYPE_MODE")
credit_train[factor_list] <- lapply(credit_train[factor_list],factor)

summary(credit_train)
#A lot of NA values present.  View NA counts by row to see if any rows should be eliminated.
credit_train$na_count <- apply(is.na(credit_train),1,sum)
ggplot(data=credit_train,aes(na_count))+geom_histogram(stat="count")

#FLAG_MOBIL has only one zero value and the remainder are 1 so this can be removed along with the temporary na_count column
credit_train <- subset(credit_train, select= -c(na_count,FLAG_MOBIL))

#create table to show NA count and percentage by column
na_table <- matrix(nrow=121, ncol=3)
colnames(na_table) <- c("variable","na_count","na_percent")

for (i in 1:121) {
  na_ct <- sum(is.na(credit_train[,i]))
  na_pct <- round(na_ct/307511,2)
  na_table[i,1] <- colnames(credit_train[i])
  na_table[i,2] <- na_ct
  na_table[i,3] <- na_pct
}
na_table <- as.data.frame(na_table)
na_table <- arrange(na_table,desc(na_percent))
View(na_table)

#Imputing will be difficult with 44 columns having 50% or more NA values.  Correlation will be used to eliminate unnecessary variables
#First, the DAYS_EMPLOYED variable looked strange in the summary with negative numbers being expected but a max value of postive 365,245.  That will be explored more.
sum(credit_train["DAYS_EMPLOYED"]>0)

#use a boxplot to see the distribution of the variable and gain insight into the positive values
ggplot(data=credit_train,aes(TARGET,DAYS_EMPLOYED))+geom_boxplot()
sum(credit_train["DAYS_EMPLOYED"]==365243)
#365243 seems to be some kind of a default value, these will be converted to NA
new_nas <- which(credit_train["DAYS_EMPLOYED"]==365243)
credit_train[new_nas,"DAYS_EMPLOYED"] <- NA

#Create a list of the factors with multiple levels that will need to be one-hot encoded and a separate list of factors with two levels
one_hot <- c("NAME_TYPE_SUITE","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY","WEEKDAY_APPR_PROCESS_START","HOUR_APPR_PROCESS_START","ORGANIZATION_TYPE","FONDKAPREMONT_MODE","HOUSETYPE_MODE","WALLSMATERIAL_MODE")
zero_one <- factor_list[! factor_list %in% one_hot]

#create a new data frame with the one-hot encoded variables
credit_train <- as.data.frame(credit_train) #convert credit_train to data frame
credit_train2 <- dummy.data.frame(credit_train, names = one_hot, sep = ".")

#Verify that all factors that should have only two levels do have only two levels
colnumbers <- NA
for (i in 1:length(zero_one)) {
  x <- which(colnames(credit_train2) == zero_one[i])
  colnumbers <- c(colnumbers,x)
}
colnumbers <- colnumbers[-1]

credit_train2 %>%
  sapply(levels)
#CODE_GENDER has more than two levels, it will need to be explored
summary.factor(credit_train2$CODE_GENDER)
#The extra level is "XNA" which should be NA.  That will be converted here
xna <- which(credit_train2$CODE_GENDER == "XNA")
credit_train2[xna,"CODE_GENDER"] <- NA
credit_train2$CODE_GENDER <- factor(credit_train2$CODE_GENDER)
summary.factor(credit_train2$CODE_GENDER)

#Convert the two-level factors to numeric 0 or 1, the as.numeric() function returns 1 or 2 so subtracting 1 will get it to 0 and 1.  
for (i in colnumbers) {
  credit_train2[[i]] <- credit_train2[[i]] %>%
    as.numeric() %>%
    -1
}

credit_train2 <- sapply(credit_train2, as.numeric)

#create a correlation matrix from the now numeric data frame
cormat <- cor(credit_train2, use = "pairwise.complete.obs")

#plot the matrix to see if anything can be gained from the visual
colpalette <- colorRampPalette(c("blue","white","red"))(20)
heatmap(cormat,col=colpalette, na.rm=TRUE, symm=TRUE)

#hard to see the correlations and high correlation is expected from the one-hot columns.  Isolate the correlation to the target variable
target_corr <- cor(x=credit_train2[,-2],y=credit_train2[,2], use = "pairwise.complete.obs") %>%
  as.data.frame()
target_corr <- target_corr[order(target_corr[,"V1"]), , drop=FALSE]
head(target_corr)
tail(target_corr)

#since there are both negative and positive correlations, take the absolute value then sort to find the most and least correlated columns
target_corr_abs <- abs(target_corr)
target_corr_abs <- target_corr_abs[order(target_corr_abs[,"V1"]), , drop=FALSE]
head(target_corr_abs,15)

#Organization_type has six of the 15 least correlated columns so it can be safely removed
elim <- "ORGANIZATION_TYPE"
i <- substr(row.names(target_corr_abs), 0, nchar(elim))==elim
i <- grepl(paste0("^", elim), row.names(target_corr_abs))
elim_nb <- which(i)
target_corr_abs <- target_corr_abs[-elim_nb, , drop=FALSE]
head(target_corr_abs,15)

#now HOUR_APPR_PROCESS_START has 5 of the bottom 15 so it will be removed
elim <- "HOUR_APPR_PROCESS_START"
i <- substr(row.names(target_corr_abs), 0, nchar(elim))==elim
i <- grepl(paste0("^", elim), row.names(target_corr_abs))
elim_nb <- which(i)
target_corr_abs <- target_corr_abs[-elim_nb, , drop=FALSE]
head(target_corr_abs,15)

#comparing the variables with 60% or more NA values (top 17 in na_table) with the target correlation to see if they can be safely removed
head(na_table, 17)
tail(target_corr_abs,30)

#None of the 17 appear in the top 30 most correlated variables so they will all be removed.  In addition, flag_document_20, flag_document_5, flag_document_12, flag_document_19, flag_document_10, flag_document_7 and flag_document_4 can all be removed due to low correlation
elim_variables <- c("ORGANIZATION_TYPE","HOUR_APPR_PROCESS_START","FLAG_DOCUMENT_4","FLAG_DOCUMENT_5","FLAG_DOCUMENT_7","FLAG_DOCUMENT_10","FLAG_DOCUMENT_12","FLAG_DOCUMENT_19","FLAG_DOCUMENT_20","COMMONAREA_AVG","COMMONAREA_MODE","COMMONAREA_MEDI","NONLIVINGAPARTMENTS_AVG","NONLIVINGAPARTMENTS_MODE","NONLIVINGAPARTMENTS_MEDI","FLOORSMIN_AVG","LIVINGAPARTMENTS_AVG","FLOORSMIN_MODE","LIVINGAPARTMENTS_MODE","FLOORSMIN_MEDI","LIVINGAPARTMENTS_MEDI","OWN_CAR_AGE","YEARS_BUILD_AVG","YEARS_BUILD_MODE","YEARS_BUILD_MEDI","FLAG_CONT_MOBILE","AMT_REQ_CREDIT_BUREAU_WEEK","AMT_REQ_CREDIT_BUREAU_HOUR")
credit_train3 <- credit_train[, -which(names(credit_train) %in% elim_variables)]

#creating a new NA table and adding the class of the variable to assist in deciding which variables to impute
na_table2 <- matrix(nrow=95, ncol=6)
colnames(na_table2) <- c("variable","na_count","na_percent", "class", "level ct", "index")

for (i in 1:95) {
  na_ct <- sum(is.na(credit_train3[,i]))
  na_pct <- round(na_ct/307511,5)
  na_table2[i,1] <- colnames(credit_train3[i])
  na_table2[i,2] <- na_ct
  na_table2[i,3] <- na_pct
  na_table2[i,4] <- class(credit_train3[,i])
  na_table2[i,5] <- length(levels(credit_train3[,i]))
  na_table2[i,6] <- i
}
na_table2 <- as.data.frame(na_table2)
na_table2 <- arrange(na_table2,desc(na_percent))
View(na_table2)

#Of the remaining 50 columns with NA values, six are factors. A new level of "None" will be added and will replace NA.  
adjust_factors <- c(69,72,70,73,26,12)
for (i in adjust_factors) {
  credit_train3[,i] <- factor(credit_train3[,i], levels=c(levels(credit_train3[,i]), "None"))
  credit_train3[is.na(credit_train3[,i]),i] <- "None"
}

#44 variables with NA values is too many for imputation so additional columns will need to be removed.  Creating a new na table with additional columns for this purpose
na_table3 <- matrix(nrow=95, ncol=6)
colnames(na_table3) <- c("variable","na_count","na_percent", "class", "level ct", "index")

for (i in 1:95) {
  na_ct <- sum(is.na(credit_train3[,i]))
  na_pct <- round(na_ct/307511,5)
  na_table3[i,1] <- colnames(credit_train3[i])
  na_table3[i,2] <- na_ct
  na_table3[i,3] <- na_pct
  na_table3[i,4] <- class(credit_train3[,i])
  na_table3[i,5] <- length(levels(credit_train3[,i]))
  na_table3[i,6] <- i
}
na_table3 <- as.data.frame(na_table3)
na_table3 <- arrange(na_table3,desc(na_percent))
View(na_table3)

#The column called DAYS_LAST_PHONE_CHANGE has only one NA value so that will be changed to a zero
ind <- which(is.na(credit_train3$DAYS_LAST_PHONE_CHANGE))
credit_train3$DAYS_LAST_PHONE_CHANGE[ind] <- 0

#All columns with more than 40% NA will be removed with the exception of EXT_SOURCE_1 because of very high correlation with the target variable
elim_variables <- c("LANDAREA_AVG","LANDAREA_MODE","LANDAREA_MEDI","BASEMENTAREA_AVG","BASEMENTAREA_MODE","BASEMENTAREA_MEDI","NONLIVINGAREA_AVG","NONLIVINGAREA_MODE","NONLIVINGAREA_MEDI","ELEVATORS_AVG","ELEVATORS_MODE","ELEVATORS_MEDI","APARTMENTS_AVG","APARTMENTS_MODE","APARTMENTS_MEDI","ENTRANCES_AVG","ENTRANCES_MODE","ENTRANCES_MEDI","LIVINGAREA_AVG","LIVINGAREA_MODE","LIVINGAREA_MEDI","FLOORSMAX_AVG","FLOORSMAX_MODE","FLOORSMAX_MEDI","YEARS_BEGINEXPLUATATION_AVG","YEARS_BEGINEXPLUATATION_MODE","YEARS_BEGINEXPLUATATION_MEDI","TOTALAREA_MODE")
credit_train3 <- credit_train3[, -which(names(credit_train3) %in% elim_variables)]

#Imputing using the cart (classification and regression trees) method
imp <- mice(credit_train3[,-2],method="cart",m=1, maxit=3)
credit_train4 <- complete(imp,1)
credit_train4$TARGET <- credit_train3$TARGET

#verify there are no NA values and the summary still looks within reason
sum(is.na(credit_train4))
summary(credit_train4)

#The data is now ready to create a models.  First, a random forest model is trained using the randomForest package

#set seed and create a train/test 70/30 split of the data
set.seed(1234)
train_ind <- sample(2,nrow(credit_train4),replace=TRUE,prob=c(0.7,0.3))
rf_train <- credit_train4[train_ind==1,]
rf_test <- credit_train4[train_ind==2,]

#create randomForest model
credit_rf <- randomForest(TARGET~., data = rf_train, importance = T, nodesize=1, ntree=100)

#Create Naive Bayes Model
nb_train <- credit_train4[train_ind==1, -65]
nb_test <- credit_train4[train_ind==2, -65]
nb_train_labels <- credit_train4[train_ind==1, 65]
nb_test_labels <- credit_train4[train_ind==2, 65]
credit_nb <- naiveBayes(nb_train, nb_train_labels, laplace = 1)

#Create dummy variables for factors and then build KNN model
factor_list4 <- c("NAME_CONTRACT_TYPE","CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY","NAME_TYPE_SUITE","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","FLAG_EMP_PHONE","FLAG_WORK_PHONE","FLAG_PHONE","FLAG_EMAIL","OCCUPATION_TYPE","REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY","WEEKDAY_APPR_PROCESS_START","REG_REGION_NOT_LIVE_REGION","REG_REGION_NOT_WORK_REGION","LIVE_REGION_NOT_WORK_REGION","REG_CITY_NOT_LIVE_CITY","REG_CITY_NOT_WORK_CITY","LIVE_CITY_NOT_WORK_CITY","FONDKAPREMONT_MODE","HOUSETYPE_MODE","WALLSMATERIAL_MODE","EMERGENCYSTATE_MODE","FLAG_DOCUMENT_2","FLAG_DOCUMENT_3","FLAG_DOCUMENT_6","FLAG_DOCUMENT_8","FLAG_DOCUMENT_9","FLAG_DOCUMENT_11","FLAG_DOCUMENT_13","FLAG_DOCUMENT_14","FLAG_DOCUMENT_15","FLAG_DOCUMENT_16","FLAG_DOCUMENT_17","FLAG_DOCUMENT_18","FLAG_DOCUMENT_21","TARGET")

#List of factors to one-hot encode
one_hot4 <- c("CODE_GENDER","NAME_TYPE_SUITE","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY","WEEKDAY_APPR_PROCESS_START","FONDKAPREMONT_MODE","HOUSETYPE_MODE","WALLSMATERIAL_MODE","EMERGENCYSTATE_MODE")

#List of factors with two levels
zero_one4 <- factor_list4[! factor_list4 %in% one_hot4]

#one-hot encode necessary factors
credit_train4 <- as.data.frame(credit_train4) #convert credit_train to data frame
credit_train5 <- dummy.data.frame(credit_train4, names = one_hot4, sep = ".")

#Column numbers for factors with two levels
colnumbers4 <- NA
for (i in 1:length(zero_one4)) {
  x <- which(colnames(credit_train5) == zero_one4[i])
  colnumbers4 <- c(colnumbers4,x)
}
colnumbers4 <- colnumbers4[-1]

#Convert two-level factors to numeric
for (i in colnumbers4) {
  credit_train5[[i]] <- credit_train5[[i]] %>%
    as.numeric() %>%
    -1
}

#Create normalize function to normalize the data, which is a requirement for the KNN model
normalize <- function(x) {
  return((x-min(x)) / (max(x)-min(x)))
}
credit_train5 <- as.data.frame(sapply(credit_train5,normalize))

#Create the train/test splits with labels separate
knn_train <- credit_train5[train_ind==1,]
knn_test <- credit_train5[train_ind==2,]
knn_train_labels <- credit_train5[train_ind==1, 139]
knn_test_labels <- credit_train5[train_ind==2, 139]
#create knn model using k=1
credit_knn <- knn(train=knn_train, test=knn_test, cl=knn_train_labels, k=1)

#Create predictions and test each model that has been created
#randomForest test
rf_prediction <- predict(credit_rf,rf_test)
confusionMatrix(table(rf_prediction, rf_test$TARGET))
#Naive Bayes test
nb_prediction <- predict(credit_nb, nb_test)
confusionMatrix(table(nb_prediction, nb_test_labels))
#KNN test
confusionMatrix(table(credit_knn,knn_test_labels))

#test the formula for balanced accuracy for the knn model (best performing)
((sum(credit_knn ==0 & knn_test_labels ==0) / sum(knn_test_labels ==0)) + (sum(credit_knn==1 & knn_test_labels ==1) / sum(knn_test_labels ==1)))/2
#create a matrix with k and the balanced accuracy to find the best value for k
acc.mat <- matrix(nrow = 10, ncol = 2)
colnames(acc.mat) <- c("k","balanced_accuracy")
for (i in 1:3) {
  pred <- knn(train = knn_train, test = knn_test, cl = knn_train_labels, k = i)
  acc <- ((sum(pred ==0 & knn_test_labels ==0) / sum(knn_test_labels ==0)) + (sum(pred==1 & knn_test_labels ==1) / sum(knn_test_labels ==1)))/2
  acc.mat[i,] = c(i,acc)
}
for (i in 4:6) {
  pred <- knn(train = knn_train, test = knn_test, cl = knn_train_labels, k = i)
  acc <- ((sum(pred ==0 & knn_test_labels ==0) / sum(knn_test_labels ==0)) + (sum(pred==1 & knn_test_labels ==1) / sum(knn_test_labels ==1)))/2
  acc.mat[i,] = c(i,acc)
}
for (i in 7:10) {
  pred <- knn(train = knn_train, test = knn_test, cl = knn_train_labels, k = i)
  acc <- ((sum(pred ==0 & knn_test_labels ==0) / sum(knn_test_labels ==0)) + (sum(pred==1 & knn_test_labels ==1) / sum(knn_test_labels ==1)))/2
  acc.mat[i,] = c(i,acc)
}

#Plot the matrix to visualize the accuracy at each k value
acc.df <- as.data.frame(acc.mat)
ggplot(acc.df, aes(x = k, y = balanced_accuracy)) + geom_line(color = "red") + geom_text(aes(label = k), hjust=0, vjust = 0, color = "blue") + labs(x = "k Value", y = "Accuracy of KNN", title = "Balanced Accuracy of KNN on Credit Risk Data")