#rm(list=ls());gc()
setwd("D:/PostDoc work/Figures/classification/Simulation/code/Github code")
source('Utilities.R')
source('POTD_utility.R')
library(e1071)
library(dr)
#library(Rdimtools)
library(pbapply)
library(class)
library(randomForest)
#

par(mar = c(4, 4, 2, 0.5)) 
#


###################################################################
##################################################################
### Hill-Valley data application ----------------------------------------
###################################################################
##################################################################

data<- read.csv("hill-valley_csv.csv")
table(data$Class)
data$y<-data$Class
data <- subset(data, select = -Class)

data$y<-as.factor(data$y)
n.size<- length(data$y)
##train and test split
train_test_splitt<- train_test_split(X=data[, -ncol(data)], y= data[, ncol(data)], test_size = 0.3, seed = 123)
train_data<- cbind(train_test_splitt$X_train,y= train_test_splitt$y_train )
test_data<- cbind(train_test_splitt$X_test,y= train_test_splitt$y_test ) 

k=round(sqrt(NROW(train_data[, ncol(train_data)])))  + (round(sqrt(NROW(train_data[, ncol(train_data)])))  %% 2 == 0)
coef.mat_logistic<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda = 0, weights = FALSE,k=k )
lambda_min<-cv.lambda_class_kk(data=train_data,weights = FALSE, k=k);lambda_min
coef.mat_lasso<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda =lambda_min , weights = FALSE, k=k)

#
df_logistic <- data.frame(coef.mat_logistic)
df_lasso <- data.frame(coef.mat_lasso)
# # Count the number of non-zero values in each column
count_nonzero <- function(column) {
  count <- sum(column != 0)
  return(count)
}
# 
counts_logistic <- sapply(df_logistic, count_nonzero) # Apply the count_nonzero function to each column of the data frame
variables <- colnames(df_logistic)
# # Plot the counts as a line plot
plot(counts_logistic, type = "o", xlab = "Predictors", ylab = "Count",
     main = "Counts of non-zero coefficients", xaxt = "n", ylim=c(0,n.size/2))
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
lines(counts_logistic, type = "o", xlab = "Predictors", ylab = "Count",
      main = "Counts of non-zero coefficients", xaxt = "n", ylim=c(0,n.size/2))
axis(1, at = 1:length(variables), labels = variables)
###Lasso------------------
counts_lasso <- sapply(df_lasso, count_nonzero) # Apply the count_nonzero function to each column of the data frame
variables <- colnames(df_lasso)
lines(counts_lasso,type = "o", col="red")



legend("topleft", 
       legend = c(expression(paste("LLO(", lambda, "= 0)")), 
                  expression(paste("LLO(", lambda, "> 0)"))),
       col = c("black", "red"), lty = 1, pch = 1)


#right singular vector-----
svd_logistic <- svd(coef.mat_logistic)
svd_lasso <- svd(coef.mat_lasso)
#
Vk_logistic <- svd_logistic$v
Vk_lasso <- svd_lasso$v

#data projection to full new subspace 
x_train_transformed_logistic_full<-as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic  
colnames(x_train_transformed_logistic_full) <- paste0("x", 1:ncol(x_train_transformed_logistic_full))
train_transformed_logistic_full<- data.frame(x_train_transformed_logistic_full, y=train_data$y)

x_test_transformed_logistic_full<-as.matrix(test_data[, -ncol(test_data)]) %*% Vk_logistic
colnames(x_test_transformed_logistic_full) <- paste0("x", 1:ncol(x_test_transformed_logistic_full))
test_transformed_logistic_full<- data.frame(x_test_transformed_logistic_full, y=test_data$y)

#lasso data------------------
x_train_transformed_lasso_full<-as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
colnames(x_train_transformed_lasso_full) <- paste0("x", 1:ncol(x_train_transformed_lasso_full))
train_transformed_lasso_full<- data.frame(x_train_transformed_lasso_full, y=train_data$y)

x_test_transformed_lasso_full<-as.matrix(test_data[, -ncol(test_data)]) %*% Vk_lasso
colnames(x_test_transformed_lasso_full) <- paste0("x", 1:ncol(x_test_transformed_lasso_full))
test_transformed_lasso_full<- data.frame(x_test_transformed_lasso_full, y=test_data$y)

###Dimension selection----------------------------------------------------------------

dimensions_logistic<-ncomp_selection3(traindata=train_transformed_logistic_full, testdata=test_transformed_logistic_full, method=c("knn"),cv=TRUE)
dimensions_lasso<-ncomp_selection3(traindata=train_transformed_lasso_full, testdata=test_transformed_lasso_full, method=c("knn"),cv=TRUE)
plot(dimensions_logistic, type="l",ylim=c(min(dimensions_lasso),0.9) ,ylab="Accuracy", xlab="Dimensions", main="Dimension selection for HV data", lwd=2 )
 #For grid---------------------
 rect(par("usr")[1], par("usr")[3],
      par("usr")[2], par("usr")[4],
      col = "#ebebeb")
 # Add white grid
 grid(nx = NULL, ny = NULL,
      col = "white", lwd = 1)
 
 lines(dimensions_logistic, type="l", lwd=2 , col="black")
 desired_components_logistic <- which.max(dimensions_logistic)[1]
 abline(v=desired_components_logistic, lty=2, col="black")
#lasso
lines(dimensions_lasso, type="l" , lwd=2, col="red")
desired_components_lasso <- which.max(dimensions_lasso)[1]
abline(v=desired_components_lasso, lty=2, col="red")
legend("topleft", 
       legend = c(expression(paste("LLO(", lambda, "= 0)")), 
                  expression(paste("LLO(", lambda, "> 0)"))),
       col = c("black", "red"), lty = 1, lwd = 2,  bty='n', inset=c(0.65, 0.01))

 
 
#
Vk_logistic <- svd_logistic$v[, 1:desired_components_logistic]
Vk_lasso <- svd_lasso$v[, 1:desired_components_lasso]
######Transform for logistic and Lasso 
x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
colnames(x_train_transformed_logistic) <- paste0("PC", 1:desired_components_logistic)
colnames(x_train_transformed_lasso) <- paste0("PC", 1:desired_components_lasso)
colnames(x_test_transformed_logistic) <- paste0("PC", 1:desired_components_logistic)
colnames(x_test_transformed_lasso) <- paste0("PC", 1:desired_components_lasso)
# mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])


###KNN-------------------------------------------------------------------
k_range <- floor(sqrt(length(train_data$y))) # Example range of k values: 1, 3, 5, 7, 9

start.time <- Sys.time()
#knn_full <- class::knn(train = train_data[,-ncol(train_data)], test = test_data[,-ncol(test_data)],cl =train_data$y,k=k_range)
knn_full <- train(x = train_data[,-ncol(train_data)], y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_full <- round(end.time - start.time,2)
#knn_full_prob <- class::knn(train = train_data[,-ncol(train_data)], test = test_data[,-ncol(test_data)],cl =train_data$y,k=k_range, prob = TRUE)

#
start.time <- Sys.time()
#knn_logistic <- class::knn(train = x_train_transformed_logistic, test = x_test_transformed_logistic,cl =train_data$y,k=k_range)
knn_logistic <-train(x =x_train_transformed_logistic, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_logistic <- round(end.time - start.time,2)
#knn_logistic_prob <- class::knn(train = x_train_transformed_logistic, test = x_test_transformed_logistic,cl =train_data$y,k=k_range, prob = TRUE)

#
start.time <- Sys.time()
#knn_lasso <- class::knn(train = x_train_transformed_lasso, test = x_test_transformed_lasso,cl =train_data$y,k=k_range)
knn_lasso <- train(x =x_train_transformed_lasso, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_lasso <- round(end.time - start.time,2)
#knn_lasso_prob <- class::knn(train = x_train_transformed_lasso, test = x_test_transformed_lasso,cl =train_data$y,k=k_range, prob = TRUE)

#
start.time <- Sys.time()
#knn_logistic_full <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range)
knn_logistic_full <- train(x =x_train_transformed_logistic_full, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_logistic_full<- round(end.time - start.time,2)
#knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)


#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_lasso_full <- train(x =x_train_transformed_lasso_full, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_lasso_full <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)

#

####
knn_time.taken<- c(time.taken_full,time.taken_logistic,time.taken_lasso,time.taken_logistic_full,time.taken_lasso_full)
names(knn_time.taken)<- c("full data model", "logistic", "lasso", "logistic_full", "lasso_full")
print(knn_time.taken)

# knn_bestTune<- c(knn_full$bestTune,knn_logistic$bestTune,knn_lasso$bestTune,knn_logistic_full$bestTune, knn_lasso_full$bestTune)
# names(knn_bestTune)<- c("k: full data model", "k: logistic", "k: lasso","k: logistic_full", "k: lasso_full")
# print(knn_bestTune)

predict_knn_full <- predict(knn_full, newdata = test_data[,-ncol(test_data)])
predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
#
predict_knn_logistic_full <- predict(knn_logistic_full, newdata = x_test_transformed_logistic_full)
predict_knn_lasso_full<- predict(knn_lasso_full, newdata = x_test_transformed_lasso_full)
#
conf_knn_full<-confusionMatrix(as.factor(predict_knn_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")

#
conf_knn_logistic_full<-confusionMatrix(as.factor(predict_knn_logistic_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_lasso_full<-confusionMatrix(as.factor(predict_knn_lasso_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")


#Use when use "knn" function estimate the model
# conf_knn_full<-confusionMatrix(as.factor(knn_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
# conf_knn_logistic<-confusionMatrix(as.factor(knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
# conf_knn_lasso<-confusionMatrix(as.factor(knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
# 
# #
# conf_knn_logistic_full<-confusionMatrix(as.factor(knn_logistic_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
# conf_knn_lasso_full<-confusionMatrix(as.factor(knn_lasso_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")

#F1 score-----------------------
# F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
# names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(F1_knn,3)
#accuracy_knn-----------------------------
# accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
# names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(accuracy_knn,3)
#AM risk------------------------------------------
AM_knn_full<-(1/2)*((1-conf_knn_full$byClass[1])+(1-conf_knn_full$byClass[2]))
AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
AM_knn_logistic_full<- (1/2)*((1-conf_knn_logistic_full$byClass[1])+(1-conf_knn_logistic_full$byClass[2]))
AM_knn_lasso_full<- (1/2)*((1-conf_knn_lasso_full$byClass[1])+(1-conf_knn_lasso_full$byClass[2]))
AM_knn<- c(AM_knn_full,AM_knn_logistic,AM_knn_lasso, AM_knn_logistic_full, AM_knn_lasso_full)
names(AM_knn)<-  c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
round(AM_knn,3)
##
#Missclassification  rate------------------------------------------
MC_knn_full<-1 - conf_knn_full$overall[1]
MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
MC_knn_lasso<-1- conf_knn_lasso$overall[1]
MC_knn_logistic_full<-1 - conf_knn_logistic_full$overall[1]
MC_knn_lasso_full<-1- conf_knn_lasso_full$overall[1]
MC_knn<- c(MC_knn_full,MC_knn_logistic,MC_knn_lasso,MC_knn_logistic_full,MC_knn_lasso_full)
names(MC_knn)<-  c("Full data model", "DR via Logistic","DR via Lasso","DR via Logistic Full","DR via Lasso Full")
round(MC_knn,3)


#KNN------------------------------------------
pred_knn_full <- predict(knn_full, newdata = test_data[,-ncol(test_data)], type = "prob")[, 2]
pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
#
pred_knn_logistic_full <- predict(knn_logistic_full, newdata = x_test_transformed_logistic_full, type = "prob")[, 2]
pred_knn_lasso_full <- predict(knn_lasso_full, newdata = x_test_transformed_lasso_full, type = "prob")[, 2]

#KNN----------------------------
pred_knn_full <- prediction(pred_knn_full, test_data[,ncol(test_data)])
pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
#
pred_knn_logistic_full <- prediction(pred_knn_logistic_full, test_data[,ncol(test_data)])
pred_knn_lasso_full <- prediction(pred_knn_lasso_full, test_data[,ncol(test_data)])
# pred_knn_full <- prediction(attr(knn_full_prob, "prob"), test_data[,ncol(test_data)])
# pred_knn_logistic <- prediction(attr(knn_logistic_prob, "prob"), test_data[,ncol(test_data)])
# pred_knn_lasso <- prediction(attr(knn_lasso_prob, "prob"), test_data[,ncol(test_data)])
# #
# pred_knn_logistic_full <- prediction(attr(knn_logistic_full_prob, "prob"), test_data[,ncol(test_data)])
# pred_knn_lasso_full <- prediction(attr(knn_lasso_full_prob, "prob"), test_data[,ncol(test_data)])
#
#perf_logis  <- performance(pred_logis, "tpr", "fpr" )


#KNN--------------------------
perf_knn_full  <- performance(pred_knn_full, "tpr", "fpr")
perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
#
perf_knn_logistic_full  <- performance(pred_knn_logistic_full, "tpr", "fpr")
perf_knn_lasso_full  <- performance(pred_knn_lasso_full, "tpr", "fpr")

#Knn----
plot(perf_knn_full,colorize = FALSE, col="black", main="", lty=1, lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
plot(perf_knn_full,colorize = FALSE, add=TRUE,col="black", main="ROC curves for models fitted through knn", lty=1, lwd=3)
plot(perf_knn_logistic,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=3)
plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="blue", lty=1,lwd=3)
# plot(perf_knn_logistic_full,add=TRUE, colorize = FALSE, col="skyblue",lty=1,lwd=3)
# plot(perf_knn_lasso_full,add=TRUE, colorize = FALSE, col="green", lty=1,lwd=3)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
legend("bottomright",legend = c("Full model", expression(paste("LLO(", lambda, "= 0)")), 
                                expression(paste("LLO(", lambda, "> 0)"))),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=3)




#Area under the curve for knn------------------------------------------------------------------
AUC_knn<-c(Auc_full=performance(pred_knn_full, "auc")@y.values[[1]], AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
           AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]])

AUC_knn

#-------------------------------------------------------------------------------------------------------------------
# paste(round(c(F1_rf[1], accuracy_rf[1],rf_time.taken[1], AM_rf[1], F1_knn[1], accuracy_knn[1],knn_time.taken[1], AM_knn[1]),3), collapse = " & ")
# paste(round(c(F1_rf[2], accuracy_rf[2],rf_time.taken[2], AM_rf[2], F1_knn[2], accuracy_knn[2],knn_time.taken[2], AM_knn[2]),3), collapse = " & ")
#   paste(round(c(F1_rf[3], accuracy_rf[3],rf_time.taken[3], AM_rf[3], F1_knn[3], accuracy_knn[3],knn_time.taken[3], AM_knn[3]),3), collapse = " & ")
# paste(round(c(F1_rf[4], accuracy_rf[4],rf_time.taken[4], AM_rf[4], F1_knn[4], accuracy_knn[4],knn_time.taken[4], AM_knn[4]),3), collapse = " & ")
# paste(round(c(F1_rf[5], accuracy_rf[5],rf_time.taken[5], AM_rf[5], F1_knn[5], accuracy_knn[5],knn_time.taken[5], AM_knn[5]),3), collapse = " & ")
# 


paste(round(c( AM_knn[1], MC_knn[1], AUC_knn[1],knn_time.taken[1]),3), collapse = " & ")
paste(round(c( AM_knn[2], MC_knn[2], AUC_knn[2],knn_time.taken[2]),3), collapse = " & ")
paste(round(c( AM_knn[3], MC_knn[3], AUC_knn[3],knn_time.taken[3]),3), collapse = " & ")
paste(round(c( AM_knn[4], MC_knn[4], AUC_knn[4],knn_time.taken[4]),3), collapse = " & ")
paste(round(c( AM_knn[5], MC_knn[5], AUC_knn[5],knn_time.taken[5]),3), collapse = " & ")



#Random forest-----------------------------------------------------------------
ntree=500
#Dimention selection throug RF----------------------------
dimensions_logistic<-ncomp_selection3(traindata=train_transformed_logistic_full, testdata=test_transformed_logistic_full, method=c("rf"),cv=TRUE)
dimensions_lasso<-ncomp_selection3(traindata=train_transformed_lasso_full, testdata=test_transformed_lasso_full, method=c("rf"),cv=TRUE)
plot(dimensions_logistic, type="l",ylim=c(min(dimensions_lasso),0.9) ,ylab="Accuracy", xlab="Dimensions", main="Dimension selection for HV data", lwd=2 )
#For grid---------------------
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")
# Add white grid
grid(nx = NULL, ny = NULL,
     col = "white", lwd = 1)

lines(dimensions_logistic, type="l", lwd=2 , col="black")
desired_components_logistic <- which.max(dimensions_logistic)[1]
abline(v=desired_components_logistic, lty=2, col="black")
#lasso
lines(dimensions_lasso, type="l" , lwd=2, col="red")
desired_components_lasso <- which.max(dimensions_lasso)[1]
abline(v=desired_components_lasso, lty=2, col="red")
legend("topleft", 
       legend = c(expression(paste("LLO(", lambda, "= 0)")), 
                  expression(paste("LLO(", lambda, "> 0)"))),
       col = c("black", "red"), lty = 1, lwd = 2,  bty='n', inset=c(0.65, 0.01))



#k <-desired_components  # Number of selected components
Vk_logistic <- svd_logistic$v[, 1:desired_components_logistic]
Vk_lasso <- svd_lasso$v[, 1:desired_components_lasso]
######Transform for logistic and Lasso 
x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
colnames(x_train_transformed_logistic) <- paste0("PC", 1:desired_components_logistic)
colnames(x_train_transformed_lasso) <- paste0("PC", 1:desired_components_lasso)
colnames(x_test_transformed_logistic) <- paste0("PC", 1:desired_components_logistic)
colnames(x_test_transformed_lasso) <- paste0("PC", 1:desired_components_lasso)
# mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])



# Random Forest model-------------------------------------------------------
mtry_tune <- tuneRF(train_data[,-ncol(train_data)], train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_full<- randomForest(x = train_data[,-ncol(train_data)],y =train_data$y,    mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_full <- round(end.time - start.time,2)

#logistic----------------------------------------------
mtry_tune <- tuneRF(x_train_transformed_logistic, train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_logistic <-  randomForest(x = x_train_transformed_logistic,y =train_data$y,   mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_logistic <- round(end.time - start.time,2)
#lasso------------------------------------------------
mtry_tune <- tuneRF(x_train_transformed_lasso, train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_lasso <-randomForest(x = x_train_transformed_lasso,y =train_data$y,   mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_lasso <- round(end.time - start.time,2)
#
#logistic fully projected space-----------------------
mtry_tune <- tuneRF(x_train_transformed_logistic_full, train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_logistic_full <- randomForest(x = x_train_transformed_logistic_full,y =train_data$y,    mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_logistic_full <- round(end.time - start.time,2)
##lasso fully projected space-----------------------
mtry_tune <- tuneRF(x_train_transformed_lasso_full, train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_lasso_full <- randomForest(x = x_train_transformed_lasso_full,y =train_data$y,   mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_lasso_full <- round(end.time - start.time,2)
#
rf_time.taken<- c(time.taken_full,time.taken_logistic,time.taken_lasso,time.taken_logistic_full,time.taken_lasso_full)
names(rf_time.taken)<- c("full data model", "logistic", "lasso", "logistic_full", "lasso_full")
print(rf_time.taken)

#plot(rf_model$err.rate, lwd="2")
min_error_tree<- c(which.min(rf_full$err.rate[,1]),which.min(rf_logistic$err.rate[,1]),which.min(rf_lasso$err.rate[,1]),
                   which.min(rf_logistic_full$err.rate[,1]),which.min(rf_lasso_full$err.rate[,1]))
names(min_error_tree)<- c("full data model", "logistic", "lasso", "logistic_full", "lasso_full")
print(min_error_tree)
# plot(rf_full)
# plot(rf_logistic, add=TRUE)
# plot(rf_lasso, add=TRUE)
# rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
# legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)


par(mfrow=c(1,1))
plot(rf_full$err.rate[,1],col="black", type="l", ylim=c(0.0,1), ylab="Error",xlab="Trees", main="OBB Error", lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
lines(rf_full$err.rate[,1],col="black", type="l", ylim=c(0.0,1), ylab="Error",xlab="Trees", main="OBB Error", lwd=3)
lines(rf_logistic$err.rate[,1],col="red", type="l",lwd=3)
lines(rf_lasso$err.rate[,1],col="blue", type="l",lwd=3)
lines(rf_logistic_full$err.rate[,1],col="skyblue", type="l",lwd=3)
lines(rf_lasso_full$err.rate[,1],col="green", type="l",lwd=3)
abline(v=which.min(rf_full$err.rate[,1]), lty=2)
abline(v=which.min(rf_logistic$err.rate[,1]), lty=2, col="red")
abline(v=which.min(rf_lasso$err.rate[,1]), lty=2, col="blue")
abline(v=which.min(rf_logistic_full$err.rate[,1]), lty=2, col="skyblue")
abline(v=which.min(rf_lasso_full$err.rate[,1]), lty=2, col="green")
#legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso"),col = c("black", "red","blue"), lty = 1)
legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso"),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=3)

#
plot(rf_full$err.rate[,2],col="black", type="l", ylim=c(0.0,0.5), ylab="Error",xlab="Trees", main="Error for class 0",lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
lines(rf_full$err.rate[,2],col="black", type="l", ylim=c(0.0,0.5), ylab="Error",xlab="Trees", main="Error for class 0",lwd=3)
lines(rf_logistic$err.rate[,2],col="red", type="l",lwd=3)
lines(rf_lasso$err.rate[,2],col="blue", type="l",lwd=3)
#
lines(rf_logistic_full$err.rate[,2],col="skyblue", type="l",lwd=3)
lines(rf_lasso_full$err.rate[,2],col="green", type="l",lwd=3)

abline(v=which.min(rf_full$err.rate[,2]), lty=2)
abline(v=which.min(rf_logistic$err.rate[,2]), lty=2, col="red")
abline(v=which.min(rf_lasso$err.rate[,2]), lty=2, col="blue")
abline(v=which.min(rf_logistic_full$err.rate[,2]), lty=2, col="skyblue")
abline(v=which.min(rf_lasso_full$err.rate[,2]), lty=2, col="green")
#legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso"),col = c("black", "red","blue"), lty = 1)
legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso"),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=3)

#
plot(rf_full$err.rate[,3],col="black", type="l", ylim=c(0.2,1.7), ylab="Error",xlab="Trees", main="Error for class 1", lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
lines(rf_full$err.rate[,3],col="black", type="l", ylim=c(0.2,1.7), ylab="Error",xlab="Trees", main="Error for class 1", lwd=3)
lines(rf_logistic$err.rate[,3],col="red", type="l", lwd=3)
lines(rf_lasso$err.rate[,3],col="blue", type="l", lwd=3)
#
lines(rf_logistic_full$err.rate[,3],col="skyblue", type="l", lwd=3)
lines(rf_lasso_full$err.rate[,3],col="green", type="l", lwd=3)

abline(v=which.min(rf_full$err.rate[,3]), lty=2, col="black")
abline(v=which.min(rf_logistic$err.rate[,3]), lty=2, col="red")
abline(v=which.min(rf_lasso$err.rate[,3]), lty=2, col="blue")
abline(v=which.min(rf_logistic_full$err.rate[,3]), lty=2, col="skyblue")
abline(v=which.min(rf_lasso_full$err.rate[,3]), lty=2, col="green")
#legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso"),col = c("black", "red","blue"), lty = 1)
legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso"),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1, lwd=3)

#importance(rf_model)
#varImpPlot(rf_model)
## Look at variable importance:
#round(importance(rf_model), 1)

#########_--------------------------------------------------------------------------------
# Predict the response using the transformed test data
predict_rf_full <- predict(rf_full, newdata= test_data[,-ncol(test_data)], type="response")
predict_rf_logistic <- predict(rf_logistic, newdata= x_test_transformed_logistic, type="response",norm.votes=TRUE)
predict_rf_lasso <- predict(rf_lasso, newdata= x_test_transformed_lasso, type="response",norm.votes=TRUE)
#
predict_rf_logistic_full <- predict(rf_logistic_full, newdata= x_test_transformed_logistic_full, type="response",norm.votes=TRUE)
predict_rf_lasso_full <- predict(rf_lasso_full, newdata= x_test_transformed_lasso_full, type="response",norm.votes=TRUE)

#Random forest---------
conf_rf_full<-confusionMatrix(as.factor(predict_rf_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_rf_logistic<-confusionMatrix(as.factor(predict_rf_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_rf_lasso<-confusionMatrix(as.factor(predict_rf_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")

conf_rf_logistic_full<-confusionMatrix(as.factor(predict_rf_logistic_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_rf_lasso_full<-confusionMatrix(as.factor(predict_rf_lasso_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")

# F1_rf<- c(conf_rf_full$byClass[7],conf_rf_logistic$byClass[7],conf_rf_lasso$byClass[7],conf_rf_logistic_full$byClass[7],conf_rf_lasso_full$byClass[7])
# names(F1_rf)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(F1_rf,3)

# 
# accuracy_rf<- c(conf_rf_full$byClass[11],conf_rf_logistic$byClass[11],conf_rf_lasso$byClass[11],conf_rf_logistic_full$byClass[11],conf_rf_lasso_full$byClass[11])
# names(accuracy_rf)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(accuracy_rf,3)
#AM risk------------------------------------------#AM risk------------------------------------------
AM_rf_full<-(1/2)*((1-conf_rf_full$byClass[1])+(1-conf_rf_full$byClass[2]))
AM_rf_logistic<-(1/2)*((1-conf_rf_logistic$byClass[1])+(1-conf_rf_logistic$byClass[2]))
AM_rf_lasso<-(1/2)*((1-conf_rf_lasso$byClass[1])+(1-conf_rf_lasso$byClass[2]))
AM_rf_logistic_full<- (1/2)*((1-conf_rf_logistic_full$byClass[1])+(1-conf_rf_logistic_full$byClass[2]))
AM_rf_lasso_full<- (1/2)*((1-conf_rf_lasso_full$byClass[1])+(1-conf_rf_lasso_full$byClass[2]))
AM_rf<- c(AM_rf_full,AM_rf_logistic,AM_rf_lasso, AM_rf_logistic_full, AM_rf_lasso_full)
names(AM_rf)<-  c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
round(AM_rf,3)
#Missclassification  rate------------------------------------------
MC_rf_full<-1 - conf_rf_full$overall[1]
MC_rf_logistic<-1 - conf_rf_logistic$overall[1]
MC_rf_lasso<-1- conf_rf_lasso$overall[1]
MC_rf_logistic_full<-1 - conf_rf_logistic_full$overall[1]
MC_rf_lasso_full<-1- conf_rf_lasso_full$overall[1]
MC_rf<- c(MC_rf_full,MC_rf_logistic,MC_rf_lasso,MC_rf_logistic_full,MC_rf_lasso_full)
names(MC_rf)<-  c("Full data model", "DR via Logistic","DR via Lasso","DR via Logistic Full","DR via Lasso Full")
round(MC_rf,3)

######################################
###ROC CURVE
# Predict probabilities for the test set (Random forest)----------------------------------
pred_rf_full <- predict(rf_full, newdata = test_data[,-ncol(test_data)], type = "prob")[, 2]
pred_rf_logistic <- predict(rf_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
pred_rf_lasso <- predict(rf_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
#
pred_rf_logistic_full <- predict(rf_logistic_full, newdata = x_test_transformed_logistic_full, type = "prob")[, 2]
pred_rf_lasso_full <- predict(rf_lasso_full, newdata = x_test_transformed_lasso_full, type = "prob")[, 2]
#

# Compute the ROC curve for logistic regression
pred_rf_full <- prediction(pred_rf_full, test_data[,ncol(test_data)])
pred_rf_logistic <- prediction(pred_rf_logistic, test_data[,ncol(test_data)])
pred_rf_lasso <- prediction(pred_rf_lasso, test_data[,ncol(test_data)])
#
pred_rf_logistic_full <- prediction(pred_rf_logistic_full, test_data[,ncol(test_data)])
pred_rf_lasso_full <- prediction(pred_rf_lasso_full, test_data[,ncol(test_data)])

#perf_logis  <- performance(pred_logis, "tpr", "fpr" )
perf_rf_full  <- performance(pred_rf_full, "tpr", "fpr")
perf_rf_logistic  <- performance(pred_rf_logistic, "tpr", "fpr")
perf_rf_lasso  <- performance(pred_rf_lasso, "tpr", "fpr")
#
perf_rf_logistic_full  <- performance(pred_rf_logistic_full, "tpr", "fpr")
perf_rf_lasso_full  <- performance(pred_rf_lasso_full, "tpr", "fpr")


#plot( perf_logis, colorize = FALSE,main = "ROC Curve")

plot(perf_rf_full, colorize = FALSE, main="",lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
plot(perf_rf_full, colorize = FALSE,add=TRUE, main="ROC curve for models fitted through RF",lwd=3)

plot(perf_rf_logistic,add=TRUE, colorize = FALSE, col="red",lwd=3)
plot(perf_rf_lasso,add=TRUE, colorize = FALSE, col="blue",lwd=3)
# plot(perf_rf_logistic_full,add=TRUE, colorize = FALSE, col="skyblue",lwd=3)
# plot(perf_rf_lasso_full,add=TRUE, colorize = FALSE, col="green",lwd=3)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

legend("bottomright",legend = c("Full model", expression(paste("LLO(", lambda, "= 0)")), 
                                expression(paste("LLO(", lambda, "> 0)")) ),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=3)


#Area under the curve Random forest------------------------------------------------------------------------

AUC_rf<-c(Auc_full=performance(pred_rf_full, "auc")@y.values[[1]], AUC_logsitc=performance(pred_rf_logistic, "auc")@y.values[[1]],
          AUC_lasso=performance(pred_rf_lasso, "auc")@y.values[[1]], AUC_logistic_full=performance(pred_rf_logistic_full, "auc")@y.values[[1]],
          AUC_lasso_full=performance(pred_rf_lasso_full, "auc")@y.values[[1]])

AUC_rf

#-------------------------------------------------------------------------------------------------------------------
# paste(round(c(F1_rf[1], accuracy_rf[1],rf_time.taken[1], AM_rf[1], F1_knn[1], accuracy_knn[1],knn_time.taken[1], AM_knn[1]),3), collapse = " & ")
# paste(round(c(F1_rf[2], accuracy_rf[2],rf_time.taken[2], AM_rf[2], F1_knn[2], accuracy_knn[2],knn_time.taken[2], AM_knn[2]),3), collapse = " & ")
#   paste(round(c(F1_rf[3], accuracy_rf[3],rf_time.taken[3], AM_rf[3], F1_knn[3], accuracy_knn[3],knn_time.taken[3], AM_knn[3]),3), collapse = " & ")
# paste(round(c(F1_rf[4], accuracy_rf[4],rf_time.taken[4], AM_rf[4], F1_knn[4], accuracy_knn[4],knn_time.taken[4], AM_knn[4]),3), collapse = " & ")
# paste(round(c(F1_rf[5], accuracy_rf[5],rf_time.taken[5], AM_rf[5], F1_knn[5], accuracy_knn[5],knn_time.taken[5], AM_knn[5]),3), collapse = " & ")
# 


paste(round(c(AM_rf[1], MC_rf[1],AUC_rf[1],rf_time.taken[1]),3), collapse = " & ") 
paste(round(c(AM_rf[2], MC_rf[2],AUC_rf[2],rf_time.taken[2]), 3), collapse = " & ")
paste(round(c(AM_rf[3], MC_rf[3],AUC_rf[3],rf_time.taken[3]), 3), collapse = " & ")
paste(round(c(AM_rf[4], MC_rf[4],AUC_rf[4],rf_time.taken[4]), 3), collapse = " & ")
paste(round(c(AM_rf[5], MC_rf[5],AUC_rf[5],rf_time.taken[5]),3), collapse = " & ")





###################################################################
##################################################################
### Wisconsin Diagnostic Brest Cancer-------------------
###################################################################
##################################################################
data<- read.csv("Breast Cancer Wisconsin (Diagnostic).csv")
table(data$diagnosis)
data$y<-data$diagnosis
data <- subset(data, select = -c(id,diagnosis,X))
data$y <- ifelse(data$y == "M", 1, data$y)
data$y <- ifelse(data$y == "B", 0, data$y)
y<- data$y
X<-data[-ncol(data)]
colnames(X) <- paste0("x", 1:length(X))
data<- cbind(X,y)
head(data)
#data<- scale(data[-ncol(data)])
data$y<-as.factor(data$y)
n.size<- length(data$y)
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
train_test_splitt<- train_test_split(X=data[, -ncol(data)], y= data[, ncol(data)], test_size = 0.3, seed = 123)

train_data<- cbind(train_test_splitt$X_train,y= train_test_splitt$y_train )
test_data<- cbind(train_test_splitt$X_test,y= train_test_splitt$y_test ) 


k=round(sqrt(NROW(train_data[, ncol(train_data)])))  + (round(sqrt(NROW(train_data[, ncol(train_data)])))  %% 2 == 0)
coef.mat_logistic<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda = 0, weights = FALSE,k=k )
lambda_min<-cv.lambda_class_kk(data=train_data,weights = FALSE, k=k);lambda_min
coef.mat_lasso<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda =lambda_min , weights = FALSE, k=k)

#
df_logistic <- data.frame(coef.mat_logistic)
df_lasso <- data.frame(coef.mat_lasso)
# # Count the number of non-zero values in each column
count_nonzero <- function(column) {
  count <- sum(column != 0)
  return(count)
}
# 
counts_logistic <- sapply(df_logistic, count_nonzero) # Apply the count_nonzero function to each column of the data frame
variables <- colnames(df_logistic)
# # Plot the counts as a line plot
plot(counts_logistic, type = "o", xlab = "Predictors", ylab = "Count",
     main = "Counts of non-zero coefficients", xaxt = "n", ylim=c(0,n.size/6))
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
lines(counts_logistic, type = "o", xlab = "Predictors", ylab = "Count",
      main = "Counts of non-zero coefficients", xaxt = "n", ylim=c(0,n.size/6))
axis(1, at = 1:length(variables), labels = variables)
###Lasso------------------
counts_lasso <- sapply(df_lasso, count_nonzero) # Apply the count_nonzero function to each column of the data frame
variables <- colnames(df_lasso)
lines(counts_lasso,type = "o", col="red")
# legend("topright", legend = c("LLO", "Lasso"),
#        col = c("black", "red","blue", "orange"), lty = 1, pch = 1)
#par( mfrow=c(1, 1)) 
# legend(10,200, legend = c(expression(paste("LLO(",lambda,"=",0)), expression(paste("LLO(", lambda,">",0))),
#        col = c("black", "red","blue", "orange"), lty = 1, pch = 1,
#        x.intersp = 0.5)



legend("topleft", 
       legend = c(expression(paste("LLO(", lambda, "= 0)")), 
                  expression(paste("LLO(", lambda, "> 0)"))),
       col = c("black", "red"), lty = 1, pch = 1)


#Right singular values----------------------------------------
svd_logistic <- svd(coef.mat_logistic)
svd_lasso <- svd(coef.mat_lasso)
#
Vk_logistic <- svd_logistic$v
Vk_lasso <- svd_lasso$v

#data for project to full new subspace-----------------------------------------------------
x_train_transformed_logistic_full<-as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
colnames(x_train_transformed_logistic_full) <- paste0("x", 1:ncol(x_train_transformed_logistic_full))
train_transformed_logistic_full<- data.frame(x_train_transformed_logistic_full, y=train_data$y)

x_test_transformed_logistic_full<-as.matrix(test_data[, -ncol(test_data)]) %*% Vk_logistic
colnames(x_test_transformed_logistic_full) <- paste0("x", 1:ncol(x_test_transformed_logistic_full))
test_transformed_logistic_full<- data.frame(x_test_transformed_logistic_full, y=test_data$y)

#lasso data------------------
x_train_transformed_lasso_full<-as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
colnames(x_train_transformed_lasso_full) <- paste0("x", 1:ncol(x_train_transformed_lasso_full))
train_transformed_lasso_full<- data.frame(x_train_transformed_lasso_full, y=train_data$y)

x_test_transformed_lasso_full<-as.matrix(test_data[, -ncol(test_data)]) %*% Vk_lasso
colnames(x_test_transformed_lasso_full) <- paste0("x", 1:ncol(x_test_transformed_lasso_full))
test_transformed_lasso_full<- data.frame(x_test_transformed_lasso_full, y=test_data$y)

#Dimension selection----------------------------
dimensions_logistic<-ncomp_selection3(traindata=train_transformed_logistic_full, testdata=test_transformed_logistic_full, method=c("knn"),cv=TRUE)
dimensions_lasso<-ncomp_selection3(traindata=train_transformed_lasso_full, testdata=test_transformed_lasso_full, method=c("knn"),cv=TRUE)
plot(dimensions_logistic, type="l",ylim=c(0.88,1) ,ylab="Accuracy", xlab="Dimensions", main="Dimension selection for HV data", lwd=2 )
#For grid---------------------
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")
# Add white grid
grid(nx = NULL, ny = NULL,
     col = "white", lwd = 1)

lines(dimensions_logistic, type="l", lwd=2 , col="black")
desired_components_logistic <- which.max(dimensions_logistic)[1]
abline(v=desired_components_logistic, lty=2, col="black")
#lasso
lines(dimensions_lasso, type="l" , lwd=2, col="red")
desired_components_lasso <- which.max(dimensions_lasso)[1]
abline(v=desired_components_lasso, lty=2, col="red")
legend("topleft", 
       legend = c(expression(paste("LLO(", lambda, "= 0)")), 
                  expression(paste("LLO(", lambda, "> 0)"))),
       col = c("black", "red"), lty = 1, lwd = 2,  bty='n', inset=c(0.65, 0.01))



#k <-desired_components  # Number of selected components
Vk_logistic <- svd_logistic$v[, 1:desired_components_logistic]
Vk_lasso <- svd_lasso$v[, 1:desired_components_lasso]
######Transform for logistic and Lasso 
x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
colnames(x_train_transformed_logistic) <- paste0("PC", 1:desired_components_logistic)
colnames(x_train_transformed_lasso) <- paste0("PC", 1:desired_components_lasso)
colnames(x_test_transformed_logistic) <- paste0("PC", 1:desired_components_logistic)
colnames(x_test_transformed_lasso) <- paste0("PC", 1:desired_components_lasso)
# mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])


###KNN-------------------------------------------------------------------
k_range <- floor(sqrt(length(train_data$y))) # Example range of k values: 1, 3, 5, 7, 9

start.time <- Sys.time()
#knn_full <- class::knn(train = train_data[,-ncol(train_data)], test = test_data[,-ncol(test_data)],cl =train_data$y,k=k_range)
knn_full <- train(x = train_data[,-ncol(train_data)], y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_full <- round(end.time - start.time,2)
#knn_full_prob <- class::knn(train = train_data[,-ncol(train_data)], test = test_data[,-ncol(test_data)],cl =train_data$y,k=k_range, prob = TRUE)

#
start.time <- Sys.time()
#knn_logistic <- class::knn(train = x_train_transformed_logistic, test = x_test_transformed_logistic,cl =train_data$y,k=k_range)
knn_logistic <-train(x =x_train_transformed_logistic, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_logistic <- round(end.time - start.time,2)
#knn_logistic_prob <- class::knn(train = x_train_transformed_logistic, test = x_test_transformed_logistic,cl =train_data$y,k=k_range, prob = TRUE)

#
start.time <- Sys.time()
#knn_lasso <- class::knn(train = x_train_transformed_lasso, test = x_test_transformed_lasso,cl =train_data$y,k=k_range)
knn_lasso <- train(x =x_train_transformed_lasso, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_lasso <- round(end.time - start.time,2)
#knn_lasso_prob <- class::knn(train = x_train_transformed_lasso, test = x_test_transformed_lasso,cl =train_data$y,k=k_range, prob = TRUE)

#
start.time <- Sys.time()
#knn_logistic_full <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range)
knn_logistic_full <- train(x =x_train_transformed_logistic_full, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_logistic_full<- round(end.time - start.time,2)
#knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)


#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_lasso_full <- train(x =x_train_transformed_lasso_full, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_lasso_full <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)

#

####
knn_time.taken<- c(time.taken_full,time.taken_logistic,time.taken_lasso,time.taken_logistic_full,time.taken_lasso_full)
names(knn_time.taken)<- c("full data model", "logistic", "lasso", "logistic_full", "lasso_full")
print(knn_time.taken)

# knn_bestTune<- c(knn_full$bestTune,knn_logistic$bestTune,knn_lasso$bestTune,knn_logistic_full$bestTune, knn_lasso_full$bestTune)
# names(knn_bestTune)<- c("k: full data model", "k: logistic", "k: lasso","k: logistic_full", "k: lasso_full")
# print(knn_bestTune)
#KNN---------------------
#KNNprediction------------------------------------------------------------------------------
predict_knn_full <- predict(knn_full, newdata = test_data[,-ncol(test_data)])
predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
#
predict_knn_logistic_full <- predict(knn_logistic_full, newdata = x_test_transformed_logistic_full)
predict_knn_lasso_full<- predict(knn_lasso_full, newdata = x_test_transformed_lasso_full)
#
conf_knn_full<-confusionMatrix(as.factor(predict_knn_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")

#
conf_knn_logistic_full<-confusionMatrix(as.factor(predict_knn_logistic_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_lasso_full<-confusionMatrix(as.factor(predict_knn_lasso_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")


#Use when use "knn" function estimate the model
# conf_knn_full<-confusionMatrix(as.factor(knn_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
# conf_knn_logistic<-confusionMatrix(as.factor(knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
# conf_knn_lasso<-confusionMatrix(as.factor(knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
# 
# #
# conf_knn_logistic_full<-confusionMatrix(as.factor(knn_logistic_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
# conf_knn_lasso_full<-confusionMatrix(as.factor(knn_lasso_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")

#F1 score-----------------------
# F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
# names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(F1_knn,3)
#accuracy_knn-----------------------------
# accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
# names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(accuracy_knn,3)
#AM risk------------------------------------------
AM_knn_full<-(1/2)*((1-conf_knn_full$byClass[1])+(1-conf_knn_full$byClass[2]))
AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
AM_knn_logistic_full<- (1/2)*((1-conf_knn_logistic_full$byClass[1])+(1-conf_knn_logistic_full$byClass[2]))
AM_knn_lasso_full<- (1/2)*((1-conf_knn_lasso_full$byClass[1])+(1-conf_knn_lasso_full$byClass[2]))
AM_knn<- c(AM_knn_full,AM_knn_logistic,AM_knn_lasso, AM_knn_logistic_full, AM_knn_lasso_full)
names(AM_knn)<-  c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
round(AM_knn,3)
##
#Missclassification  rate------------------------------------------
MC_knn_full<-1 - conf_knn_full$overall[1]
MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
MC_knn_lasso<-1- conf_knn_lasso$overall[1]
MC_knn_logistic_full<-1 - conf_knn_logistic_full$overall[1]
MC_knn_lasso_full<-1- conf_knn_lasso_full$overall[1]
MC_knn<- c(MC_knn_full,MC_knn_logistic,MC_knn_lasso,MC_knn_logistic_full,MC_knn_lasso_full)
names(MC_knn)<-  c("Full data model", "DR via Logistic","DR via Lasso","DR via Logistic Full","DR via Lasso Full")
round(MC_knn,3)



######################################
###ROC CURVE

#KNN------------------------------------------
pred_knn_full <- predict(knn_full, newdata = test_data[,-ncol(test_data)], type = "prob")[, 2]
pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
#
pred_knn_logistic_full <- predict(knn_logistic_full, newdata = x_test_transformed_logistic_full, type = "prob")[, 2]
pred_knn_lasso_full <- predict(knn_lasso_full, newdata = x_test_transformed_lasso_full, type = "prob")[, 2]

#KNN----------------------------
pred_knn_full <- prediction(pred_knn_full, test_data[,ncol(test_data)])
pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
#
pred_knn_logistic_full <- prediction(pred_knn_logistic_full, test_data[,ncol(test_data)])
pred_knn_lasso_full <- prediction(pred_knn_lasso_full, test_data[,ncol(test_data)])
# pred_knn_full <- prediction(attr(knn_full_prob, "prob"), test_data[,ncol(test_data)])
# pred_knn_logistic <- prediction(attr(knn_logistic_prob, "prob"), test_data[,ncol(test_data)])
# pred_knn_lasso <- prediction(attr(knn_lasso_prob, "prob"), test_data[,ncol(test_data)])
# #
# pred_knn_logistic_full <- prediction(attr(knn_logistic_full_prob, "prob"), test_data[,ncol(test_data)])
# pred_knn_lasso_full <- prediction(attr(knn_lasso_full_prob, "prob"), test_data[,ncol(test_data)])
#
#perf_logis  <- performance(pred_logis, "tpr", "fpr" )


#KNN--------------------------
perf_knn_full  <- performance(pred_knn_full, "tpr", "fpr")
perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
#
perf_knn_logistic_full  <- performance(pred_knn_logistic_full, "tpr", "fpr")
perf_knn_lasso_full  <- performance(pred_knn_lasso_full, "tpr", "fpr")

#Knn----
plot(perf_knn_full,colorize = FALSE, col="black", main="", lty=1, lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
plot(perf_knn_full,colorize = FALSE, add=TRUE,col="black", main="ROC curves for models fitted through knn", lty=1, lwd=3)
plot(perf_knn_logistic,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=3)
plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="blue", lty=1,lwd=3)
# plot(perf_knn_logistic_full,add=TRUE, colorize = FALSE, col="skyblue",lty=1,lwd=3)
# plot(perf_knn_lasso_full,add=TRUE, colorize = FALSE, col="green", lty=1,lwd=3)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
legend("bottomright",legend = c("Full model", expression(paste("LLO(", lambda, "= 0)")), 
                                expression(paste("LLO(", lambda, "> 0)")) ),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=3)




#Area under the curve for knn------------------------------------------------------------------
AUC_knn<-c(Auc_full=performance(pred_knn_full, "auc")@y.values[[1]], AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
           AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_logistic_full=performance(pred_knn_logistic_full, "auc")@y.values[[1]],
           AUC_lasso_full=performance(pred_knn_lasso_full, "auc")@y.values[[1]])

#-------------------------------------------------------------------------------------------------------------------
# paste(round(c(F1_rf[1], accuracy_rf[1],rf_time.taken[1], AM_rf[1], F1_knn[1], accuracy_knn[1],knn_time.taken[1], AM_knn[1]),3), collapse = " & ")
# paste(round(c(F1_rf[2], accuracy_rf[2],rf_time.taken[2], AM_rf[2], F1_knn[2], accuracy_knn[2],knn_time.taken[2], AM_knn[2]),3), collapse = " & ")
#   paste(round(c(F1_rf[3], accuracy_rf[3],rf_time.taken[3], AM_rf[3], F1_knn[3], accuracy_knn[3],knn_time.taken[3], AM_knn[3]),3), collapse = " & ")
# paste(round(c(F1_rf[4], accuracy_rf[4],rf_time.taken[4], AM_rf[4], F1_knn[4], accuracy_knn[4],knn_time.taken[4], AM_knn[4]),3), collapse = " & ")
# paste(round(c(F1_rf[5], accuracy_rf[5],rf_time.taken[5], AM_rf[5], F1_knn[5], accuracy_knn[5],knn_time.taken[5], AM_knn[5]),3), collapse = " & ")
# 


paste(round(c( AM_knn[1], MC_knn[1], AUC_knn[1],knn_time.taken[1]),3), collapse = " & ")
paste(round(c( AM_knn[2], MC_knn[2], AUC_knn[2],knn_time.taken[2]),3), collapse = " & ")
paste(round(c( AM_knn[3], MC_knn[3], AUC_knn[3],knn_time.taken[3]),3), collapse = " & ")
paste(round(c( AM_knn[4], MC_knn[4], AUC_knn[4],knn_time.taken[4]),3), collapse = " & ")
paste(round(c( AM_knn[5], MC_knn[5], AUC_knn[5],knn_time.taken[5]),3), collapse = " & ")



#Random forest-----------------------------------------------------------------
ntree=500
#Dimention selection throug RF----------------------------
dimensions_logistic<-ncomp_selection3(traindata=train_transformed_logistic_full, testdata=test_transformed_logistic_full, method=c("rf"),cv=TRUE)
dimensions_lasso<-ncomp_selection3(traindata=train_transformed_lasso_full, testdata=test_transformed_lasso_full, method=c("rf"),cv=TRUE)
plot(dimensions_logistic, type="l",ylim=c(0.88,1) ,ylab="Accuracy", xlab="Dimensions", main="Dimension selection for HV data", lwd=2 )
#For grid---------------------
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")
# Add white grid
grid(nx = NULL, ny = NULL,
     col = "white", lwd = 1)

lines(dimensions_logistic, type="l", lwd=2 , col="black")
desired_components_logistic <- which.max(dimensions_logistic)[1]
abline(v=desired_components_logistic, lty=2, col="black")
#lasso
lines(dimensions_lasso, type="l" , lwd=2, col="red")
desired_components_lasso <- which.max(dimensions_lasso)[1]
abline(v=desired_components_lasso, lty=2, col="red")
legend("topleft", 
       legend = c(expression(paste("LLO(", lambda, "= 0)")), 
                  expression(paste("LLO(", lambda, "> 0)"))),
       col = c("black", "red"), lty = 1, lwd = 2,  bty='n', inset=c(0.65, 0.01))



#k <-desired_components  # Number of selected components
Vk_logistic <- svd_logistic$v[, 1:desired_components_logistic]
Vk_lasso <- svd_lasso$v[, 1:desired_components_lasso]
######Transform for logistic and Lasso 
x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
colnames(x_train_transformed_logistic) <- paste0("PC", 1:desired_components_logistic)
colnames(x_train_transformed_lasso) <- paste0("PC", 1:desired_components_lasso)
colnames(x_test_transformed_logistic) <- paste0("PC", 1:desired_components_logistic)
colnames(x_test_transformed_lasso) <- paste0("PC", 1:desired_components_lasso)
# mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])



# Random Forest model-------------------------------------------------------
mtry_tune <- tuneRF(train_data[,-ncol(train_data)], train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_full<- randomForest(x = train_data[,-ncol(train_data)],y =train_data$y,    mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_full <- round(end.time - start.time,2)

#logistic----------------------------------------------
mtry_tune <- tuneRF(x_train_transformed_logistic, train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_logistic <-  randomForest(x = x_train_transformed_logistic,y =train_data$y,   mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_logistic <- round(end.time - start.time,2)
#lasso------------------------------------------------
mtry_tune <- tuneRF(x_train_transformed_lasso, train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_lasso <-randomForest(x = x_train_transformed_lasso,y =train_data$y,   mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_lasso <- round(end.time - start.time,2)
#
#logistic fully projected space-----------------------
mtry_tune <- tuneRF(x_train_transformed_logistic_full, train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_logistic_full <- randomForest(x = x_train_transformed_logistic_full,y =train_data$y,    mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_logistic_full <- round(end.time - start.time,2)
##lasso fully projected space-----------------------
mtry_tune <- tuneRF(x_train_transformed_lasso_full, train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_lasso_full <- randomForest(x = x_train_transformed_lasso_full,y =train_data$y,   mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_lasso_full <- round(end.time - start.time,2)
#
rf_time.taken<- c(time.taken_full,time.taken_logistic,time.taken_lasso,time.taken_logistic_full,time.taken_lasso_full)
names(rf_time.taken)<- c("full data model", "logistic", "lasso", "logistic_full", "lasso_full")
print(rf_time.taken)

#plot(rf_model$err.rate, lwd="2")
min_error_tree<- c(which.min(rf_full$err.rate[,1]),which.min(rf_logistic$err.rate[,1]),which.min(rf_lasso$err.rate[,1]),
                   which.min(rf_logistic_full$err.rate[,1]),which.min(rf_lasso_full$err.rate[,1]))
names(min_error_tree)<- c("full data model", "logistic", "lasso", "logistic_full", "lasso_full")
print(min_error_tree)
# plot(rf_full)
# plot(rf_logistic, add=TRUE)
# plot(rf_lasso, add=TRUE)
# rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
# legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)

plot(rf_full$err.rate[,1],col="black", type="l", ylim=c(0.0,1), ylab="Error",xlab="Trees", main="OBB Error", lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
lines(rf_full$err.rate[,1],col="black", type="l", ylim=c(0.0,1), ylab="Error",xlab="Trees", main="OBB Error", lwd=3)
lines(rf_logistic$err.rate[,1],col="red", type="l",lwd=3)
lines(rf_lasso$err.rate[,1],col="blue", type="l",lwd=3)
lines(rf_logistic_full$err.rate[,1],col="skyblue", type="l",lwd=3)
lines(rf_lasso_full$err.rate[,1],col="green", type="l",lwd=3)
abline(v=which.min(rf_full$err.rate[,1]), lty=2)
abline(v=which.min(rf_logistic$err.rate[,1]), lty=2, col="red")
abline(v=which.min(rf_lasso$err.rate[,1]), lty=2, col="blue")
abline(v=which.min(rf_logistic_full$err.rate[,1]), lty=2, col="skyblue")
abline(v=which.min(rf_lasso_full$err.rate[,1]), lty=2, col="green")
#legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso"),col = c("black", "red","blue"), lty = 1)
legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso"),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=3)

#
plot(rf_full$err.rate[,2],col="black", type="l", ylim=c(0.0,0.5), ylab="Error",xlab="Trees", main="Error for class 0",lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
lines(rf_full$err.rate[,2],col="black", type="l", ylim=c(0.0,0.5), ylab="Error",xlab="Trees", main="Error for class 0",lwd=3)
lines(rf_logistic$err.rate[,2],col="red", type="l",lwd=3)
lines(rf_lasso$err.rate[,2],col="blue", type="l",lwd=3)
#
lines(rf_logistic_full$err.rate[,2],col="skyblue", type="l",lwd=3)
lines(rf_lasso_full$err.rate[,2],col="green", type="l",lwd=3)

abline(v=which.min(rf_full$err.rate[,2]), lty=2)
abline(v=which.min(rf_logistic$err.rate[,2]), lty=2, col="red")
abline(v=which.min(rf_lasso$err.rate[,2]), lty=2, col="blue")
abline(v=which.min(rf_logistic_full$err.rate[,2]), lty=2, col="skyblue")
abline(v=which.min(rf_lasso_full$err.rate[,2]), lty=2, col="green")
#legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso"),col = c("black", "red","blue"), lty = 1)
legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso"),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=3)

#
plot(rf_full$err.rate[,3],col="black", type="l", ylim=c(0.2,1.7), ylab="Error",xlab="Trees", main="Error for class 1", lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
lines(rf_full$err.rate[,3],col="black", type="l", ylim=c(0.2,1.7), ylab="Error",xlab="Trees", main="Error for class 1", lwd=3)
lines(rf_logistic$err.rate[,3],col="red", type="l", lwd=3)
lines(rf_lasso$err.rate[,3],col="blue", type="l", lwd=3)
#
lines(rf_logistic_full$err.rate[,3],col="skyblue", type="l", lwd=3)
lines(rf_lasso_full$err.rate[,3],col="green", type="l", lwd=3)

abline(v=which.min(rf_full$err.rate[,3]), lty=2, col="black")
abline(v=which.min(rf_logistic$err.rate[,3]), lty=2, col="red")
abline(v=which.min(rf_lasso$err.rate[,3]), lty=2, col="blue")
abline(v=which.min(rf_logistic_full$err.rate[,3]), lty=2, col="skyblue")
abline(v=which.min(rf_lasso_full$err.rate[,3]), lty=2, col="green")
#legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso"),col = c("black", "red","blue"), lty = 1)
legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso"),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1, lwd=3)

#importance(rf_model)
#varImpPlot(rf_model)
## Look at variable importance:
#round(importance(rf_model), 1)

#########_--------------------------------------------------------------------------------
# Predict the response using the transformed test data
predict_rf_full <- predict(rf_full, newdata= test_data[,-ncol(test_data)], type="response")
predict_rf_logistic <- predict(rf_logistic, newdata= x_test_transformed_logistic, type="response",norm.votes=TRUE)
predict_rf_lasso <- predict(rf_lasso, newdata= x_test_transformed_lasso, type="response",norm.votes=TRUE)
#
predict_rf_logistic_full <- predict(rf_logistic_full, newdata= x_test_transformed_logistic_full, type="response",norm.votes=TRUE)
predict_rf_lasso_full <- predict(rf_lasso_full, newdata= x_test_transformed_lasso_full, type="response",norm.votes=TRUE)

#Random forest---------
conf_rf_full<-confusionMatrix(as.factor(predict_rf_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_rf_logistic<-confusionMatrix(as.factor(predict_rf_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_rf_lasso<-confusionMatrix(as.factor(predict_rf_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")

conf_rf_logistic_full<-confusionMatrix(as.factor(predict_rf_logistic_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_rf_lasso_full<-confusionMatrix(as.factor(predict_rf_lasso_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")

# F1_rf<- c(conf_rf_full$byClass[7],conf_rf_logistic$byClass[7],conf_rf_lasso$byClass[7],conf_rf_logistic_full$byClass[7],conf_rf_lasso_full$byClass[7])
# names(F1_rf)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(F1_rf,3)

# 
# accuracy_rf<- c(conf_rf_full$byClass[11],conf_rf_logistic$byClass[11],conf_rf_lasso$byClass[11],conf_rf_logistic_full$byClass[11],conf_rf_lasso_full$byClass[11])
# names(accuracy_rf)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(accuracy_rf,3)
#AM risk------------------------------------------#AM risk------------------------------------------
AM_rf_full<-(1/2)*((1-conf_rf_full$byClass[1])+(1-conf_rf_full$byClass[2]))
AM_rf_logistic<-(1/2)*((1-conf_rf_logistic$byClass[1])+(1-conf_rf_logistic$byClass[2]))
AM_rf_lasso<-(1/2)*((1-conf_rf_lasso$byClass[1])+(1-conf_rf_lasso$byClass[2]))
AM_rf_logistic_full<- (1/2)*((1-conf_rf_logistic_full$byClass[1])+(1-conf_rf_logistic_full$byClass[2]))
AM_rf_lasso_full<- (1/2)*((1-conf_rf_lasso_full$byClass[1])+(1-conf_rf_lasso_full$byClass[2]))
AM_rf<- c(AM_rf_full,AM_rf_logistic,AM_rf_lasso, AM_rf_logistic_full, AM_rf_lasso_full)
names(AM_rf)<-  c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
round(AM_rf,3)
#Missclassification  rate------------------------------------------
MC_rf_full<-1 - conf_rf_full$overall[1]
MC_rf_logistic<-1 - conf_rf_logistic$overall[1]
MC_rf_lasso<-1- conf_rf_lasso$overall[1]
MC_rf_logistic_full<-1 - conf_rf_logistic_full$overall[1]
MC_rf_lasso_full<-1- conf_rf_lasso_full$overall[1]
MC_rf<- c(MC_rf_full,MC_rf_logistic,MC_rf_lasso,MC_rf_logistic_full,MC_rf_lasso_full)
names(MC_rf)<-  c("Full data model", "DR via Logistic","DR via Lasso","DR via Logistic Full","DR via Lasso Full")
round(MC_rf,3)

######################################
###ROC CURVE
# Predict probabilities for the test set (Random forest)----------------------------------
pred_rf_full <- predict(rf_full, newdata = test_data[,-ncol(test_data)], type = "prob")[, 2]
pred_rf_logistic <- predict(rf_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
pred_rf_lasso <- predict(rf_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
#
pred_rf_logistic_full <- predict(rf_logistic_full, newdata = x_test_transformed_logistic_full, type = "prob")[, 2]
pred_rf_lasso_full <- predict(rf_lasso_full, newdata = x_test_transformed_lasso_full, type = "prob")[, 2]
#

# Compute the ROC curve for logistic regression
pred_rf_full <- prediction(pred_rf_full, test_data[,ncol(test_data)])
pred_rf_logistic <- prediction(pred_rf_logistic, test_data[,ncol(test_data)])
pred_rf_lasso <- prediction(pred_rf_lasso, test_data[,ncol(test_data)])
#
pred_rf_logistic_full <- prediction(pred_rf_logistic_full, test_data[,ncol(test_data)])
pred_rf_lasso_full <- prediction(pred_rf_lasso_full, test_data[,ncol(test_data)])

#perf_logis  <- performance(pred_logis, "tpr", "fpr" )
perf_rf_full  <- performance(pred_rf_full, "tpr", "fpr")
perf_rf_logistic  <- performance(pred_rf_logistic, "tpr", "fpr")
perf_rf_lasso  <- performance(pred_rf_lasso, "tpr", "fpr")
#
perf_rf_logistic_full  <- performance(pred_rf_logistic_full, "tpr", "fpr")
perf_rf_lasso_full  <- performance(pred_rf_lasso_full, "tpr", "fpr")


#plot( perf_logis, colorize = FALSE,main = "ROC Curve")

plot(perf_rf_full, colorize = FALSE, main="",lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
plot(perf_rf_full, colorize = FALSE,add=TRUE, main="ROC curve for models fitted through RF",lwd=3)

plot(perf_rf_logistic,add=TRUE, colorize = FALSE, col="red",lwd=3)
plot(perf_rf_lasso,add=TRUE, colorize = FALSE, col="blue",lwd=3)
# plot(perf_rf_logistic_full,add=TRUE, colorize = FALSE, col="skyblue",lwd=3)
# plot(perf_rf_lasso_full,add=TRUE, colorize = FALSE, col="green",lwd=3)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

legend("bottomright",legend = c("Full model", expression(paste("LLO(", lambda, "= 0)")), 
                                expression(paste("LLO(", lambda, "> 0)")) ),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=3)


#Area under the curve Random forest------------------------------------------------------------------------

AUC_rf<-c(Auc_full=performance(pred_rf_full, "auc")@y.values[[1]], AUC_logsitc=performance(pred_rf_logistic, "auc")@y.values[[1]],
          AUC_lasso=performance(pred_rf_lasso, "auc")@y.values[[1]], AUC_logistic_full=performance(pred_rf_logistic_full, "auc")@y.values[[1]],
          AUC_lasso_full=performance(pred_rf_lasso_full, "auc")@y.values[[1]])

AUC_rf

#-------------------------------------------------------------------------------------------------------------------
# paste(round(c(F1_rf[1], accuracy_rf[1],rf_time.taken[1], AM_rf[1], F1_knn[1], accuracy_knn[1],knn_time.taken[1], AM_knn[1]),3), collapse = " & ")
# paste(round(c(F1_rf[2], accuracy_rf[2],rf_time.taken[2], AM_rf[2], F1_knn[2], accuracy_knn[2],knn_time.taken[2], AM_knn[2]),3), collapse = " & ")
#   paste(round(c(F1_rf[3], accuracy_rf[3],rf_time.taken[3], AM_rf[3], F1_knn[3], accuracy_knn[3],knn_time.taken[3], AM_knn[3]),3), collapse = " & ")
# paste(round(c(F1_rf[4], accuracy_rf[4],rf_time.taken[4], AM_rf[4], F1_knn[4], accuracy_knn[4],knn_time.taken[4], AM_knn[4]),3), collapse = " & ")
# paste(round(c(F1_rf[5], accuracy_rf[5],rf_time.taken[5], AM_rf[5], F1_knn[5], accuracy_knn[5],knn_time.taken[5], AM_knn[5]),3), collapse = " & ")
# 


paste(round(c(AM_rf[1], MC_rf[1],AUC_rf[1],rf_time.taken[1]),3), collapse = " & ") 
paste(round(c(AM_rf[2], MC_rf[2],AUC_rf[2],rf_time.taken[2]), 3), collapse = " & ")
paste(round(c(AM_rf[3], MC_rf[3],AUC_rf[3],rf_time.taken[3]), 3), collapse = " & ")
paste(round(c(AM_rf[4], MC_rf[4],AUC_rf[4],rf_time.taken[4]), 3), collapse = " & ")
paste(round(c(AM_rf[5], MC_rf[5],AUC_rf[5],rf_time.taken[5]),3), collapse = " & ")





###################################################################
##################################################################
### mice+protein+expression-------------------
###################################################################
##################################################################

ntree=500
data<-read.csv("Data_Cortex_Nuclear.csv")
data<- na.omit(data)
ntree=500
data$y<-data$class
data <- subset(data, select = -class)
data$y <- ifelse(data$y == "c-CS-m", 0, data$y)
data$y <- ifelse(data$y == "c-CS-s", 0, data$y)
data$y <- ifelse(data$y == "c-SC-m", 0, data$y)
data$y <- ifelse(data$y == "c-SC-s", 0, data$y)
data$y <- ifelse(data$y != "0", 1, data$y)
table(data$y)
y<- data$y
X<-data[-ncol(data)]
colnames(X) <- paste0("x", 1:length(X))
#X<- scale(X)
data<- data.frame(X,y)

data$y<-as.factor(data$y)
n.size<- length(data$y)
##------------------------------------------------------------------------------------------------
train_test_splitt<- train_test_split(X=data[, -ncol(data)], y= data[, ncol(data)], test_size = 0.3, seed = 123)

train_data<- cbind(train_test_splitt$X_train,y= train_test_splitt$y_train )
test_data<- cbind(train_test_splitt$X_test,y= train_test_splitt$y_test ) 

k=round(sqrt(NROW(train_data[, ncol(train_data)])))  + (round(sqrt(NROW(train_data[, ncol(train_data)])))  %% 2 == 0)
coef.mat_logistic<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda = 0, weights = FALSE,k=k )
lambda_min<-cv.lambda_class_kk(data=train_data,weights = FALSE, k=k);lambda_min
coef.mat_lasso<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda =lambda_min , weights = FALSE, k=k)

#
df_logistic <- data.frame(coef.mat_logistic)
df_lasso <- data.frame(coef.mat_lasso)
# # Count the number of non-zero values in each column
count_nonzero <- function(column) {
  count <- sum(column != 0)
  return(count)
}
# 
counts_logistic <- sapply(df_logistic, count_nonzero) # Apply the count_nonzero function to each column of the data frame
variables <- colnames(df_logistic)
# # Plot the counts as a line plot
plot(counts_logistic, type = "o", xlab = "Predictors", ylab = "Count",
     main = "Counts of non-zero coefficients", xaxt = "n", ylim=c(0,n.size/3))
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
lines(counts_logistic, type = "o", xlab = "Predictors", ylab = "Count",
      main = "Counts of non-zero coefficients", xaxt = "n", ylim=c(0,n.size/3))
axis(1, at = 1:length(variables), labels = variables)
###Lasso------------------
counts_lasso <- sapply(df_lasso, count_nonzero) # Apply the count_nonzero function to each column of the data frame
variables <- colnames(df_lasso)
lines(counts_lasso,type = "o", col="red")
# legend("topright", legend = c("LLO", "Lasso"),
#        col = c("black", "red","blue", "orange"), lty = 1, pch = 1)
#par( mfrow=c(1, 1)) 
# legend(10,200, legend = c(expression(paste("LLO(",lambda,"=",0)), expression(paste("LLO(", lambda,">",0))),
#        col = c("black", "red","blue", "orange"), lty = 1, pch = 1,
#        x.intersp = 0.5)



legend("topleft", 
       legend = c(expression(paste("LLO(", lambda, "= 0)")), 
                  expression(paste("LLO(", lambda, "> 0)"))),
       col = c("black", "red"), lty = 1, pch = 1)


###############
svd_logistic <- svd(coef.mat_logistic)
svd_lasso <- svd(coef.mat_lasso)
#
Vk_logistic <- svd_logistic$v
Vk_lasso <- svd_lasso$v

#data project to full new subspace-----------------------------------------------------
x_train_transformed_logistic_full<-as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
colnames(x_train_transformed_logistic_full) <- paste0("x", 1:ncol(x_train_transformed_logistic_full))
train_transformed_logistic_full<- data.frame(x_train_transformed_logistic_full, y=train_data$y)

x_test_transformed_logistic_full<-as.matrix(test_data[, -ncol(test_data)]) %*% Vk_logistic
colnames(x_test_transformed_logistic_full) <- paste0("x", 1:ncol(x_test_transformed_logistic_full))
test_transformed_logistic_full<- data.frame(x_test_transformed_logistic_full, y=test_data$y)

#lasso data------------------
x_train_transformed_lasso_full<-as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
colnames(x_train_transformed_lasso_full) <- paste0("x", 1:ncol(x_train_transformed_lasso_full))
train_transformed_lasso_full<- data.frame(x_train_transformed_lasso_full, y=train_data$y)

x_test_transformed_lasso_full<-as.matrix(test_data[, -ncol(test_data)]) %*% Vk_lasso
colnames(x_test_transformed_lasso_full) <- paste0("x", 1:ncol(x_test_transformed_lasso_full))
test_transformed_lasso_full<- data.frame(x_test_transformed_lasso_full, y=test_data$y)

#Dimention selection----------------------------
dimensions_logistic<-ncomp_selection3(traindata=train_transformed_logistic_full, testdata=test_transformed_logistic_full, method=c("knn"),cv=TRUE)
dimensions_lasso<-ncomp_selection3(traindata=train_transformed_lasso_full, testdata=test_transformed_lasso_full, method=c("knn"),cv=TRUE)
plot(dimensions_logistic, type="l",ylim=c(0.7,1) ,ylab="Accuracy", xlab="Dimensions", main="Dimension selection for HV data", lwd=2 )
#For grid---------------------
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")
# Add white grid
grid(nx = NULL, ny = NULL,
     col = "white", lwd = 1)

lines(dimensions_logistic, type="l", lwd=2 , col="black")
desired_components_logistic <- which.max(dimensions_logistic)[1]
abline(v=desired_components_logistic, lty=2, col="black")
#lasso
lines(dimensions_lasso, type="l" , lwd=2, col="red")
desired_components_lasso <- which.max(dimensions_lasso)[1]
abline(v=desired_components_lasso, lty=2, col="red")
legend("topleft", 
       legend = c(expression(paste("LLO(", lambda, "= 0)")), 
                  expression(paste("LLO(", lambda, "> 0)"))),
       col = c("black", "red"), lty = 1, lwd = 2,  bty='n', inset=c(0.65, 0.01))


#k <-desired_components  # Number of selected components
Vk_logistic <- svd_logistic$v[, 1:desired_components_logistic]
Vk_lasso <- svd_lasso$v[, 1:desired_components_lasso]
######Transform for logistic and Lasso 
x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
colnames(x_train_transformed_logistic) <- paste0("PC", 1:desired_components_logistic)
colnames(x_train_transformed_lasso) <- paste0("PC", 1:desired_components_lasso)
colnames(x_test_transformed_logistic) <- paste0("PC", 1:desired_components_logistic)
colnames(x_test_transformed_lasso) <- paste0("PC", 1:desired_components_lasso)
# mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])


###KNN-------------------------------------------------------------------
k_range <- floor(sqrt(length(train_data$y))) # Example range of k values: 1, 3, 5, 7, 9

start.time <- Sys.time()
#knn_full <- class::knn(train = train_data[,-ncol(train_data)], test = test_data[,-ncol(test_data)],cl =train_data$y,k=k_range)
knn_full <- train(x = train_data[,-ncol(train_data)], y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_full <- round(end.time - start.time,2)
#knn_full_prob <- class::knn(train = train_data[,-ncol(train_data)], test = test_data[,-ncol(test_data)],cl =train_data$y,k=k_range, prob = TRUE)

#
start.time <- Sys.time()
#knn_logistic <- class::knn(train = x_train_transformed_logistic, test = x_test_transformed_logistic,cl =train_data$y,k=k_range)
knn_logistic <-train(x =x_train_transformed_logistic, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_logistic <- round(end.time - start.time,2)
#knn_logistic_prob <- class::knn(train = x_train_transformed_logistic, test = x_test_transformed_logistic,cl =train_data$y,k=k_range, prob = TRUE)

#
start.time <- Sys.time()
#knn_lasso <- class::knn(train = x_train_transformed_lasso, test = x_test_transformed_lasso,cl =train_data$y,k=k_range)
knn_lasso <- train(x =x_train_transformed_lasso, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_lasso <- round(end.time - start.time,2)
#knn_lasso_prob <- class::knn(train = x_train_transformed_lasso, test = x_test_transformed_lasso,cl =train_data$y,k=k_range, prob = TRUE)

#
start.time <- Sys.time()
#knn_logistic_full <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range)
knn_logistic_full <- train(x =x_train_transformed_logistic_full, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_logistic_full<- round(end.time - start.time,2)
#knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)


#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_lasso_full <- train(x =x_train_transformed_lasso_full, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_lasso_full <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)

#

####
knn_time.taken<- c(time.taken_full,time.taken_logistic,time.taken_lasso,time.taken_logistic_full,time.taken_lasso_full)
names(knn_time.taken)<- c("full data model", "logistic", "lasso", "logistic_full", "lasso_full")
print(knn_time.taken)

# knn_bestTune<- c(knn_full$bestTune,knn_logistic$bestTune,knn_lasso$bestTune,knn_logistic_full$bestTune, knn_lasso_full$bestTune)
# names(knn_bestTune)<- c("k: full data model", "k: logistic", "k: lasso","k: logistic_full", "k: lasso_full")
# print(knn_bestTune)
#KNN---------------------
#KNNprediction------------------------------------------------------------------------------
predict_knn_full <- predict(knn_full, newdata = test_data[,-ncol(test_data)])
predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
#
predict_knn_logistic_full <- predict(knn_logistic_full, newdata = x_test_transformed_logistic_full)
predict_knn_lasso_full<- predict(knn_lasso_full, newdata = x_test_transformed_lasso_full)
#
conf_knn_full<-confusionMatrix(as.factor(predict_knn_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")

#
conf_knn_logistic_full<-confusionMatrix(as.factor(predict_knn_logistic_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_lasso_full<-confusionMatrix(as.factor(predict_knn_lasso_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")


#Use when use "knn" function estimate the model
# conf_knn_full<-confusionMatrix(as.factor(knn_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
# conf_knn_logistic<-confusionMatrix(as.factor(knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
# conf_knn_lasso<-confusionMatrix(as.factor(knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
# 
# #
# conf_knn_logistic_full<-confusionMatrix(as.factor(knn_logistic_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
# conf_knn_lasso_full<-confusionMatrix(as.factor(knn_lasso_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")

#F1 score-----------------------
# F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
# names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(F1_knn,3)
#accuracy_knn-----------------------------
# accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
# names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(accuracy_knn,3)
#AM risk------------------------------------------
AM_knn_full<-(1/2)*((1-conf_knn_full$byClass[1])+(1-conf_knn_full$byClass[2]))
AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
AM_knn_logistic_full<- (1/2)*((1-conf_knn_logistic_full$byClass[1])+(1-conf_knn_logistic_full$byClass[2]))
AM_knn_lasso_full<- (1/2)*((1-conf_knn_lasso_full$byClass[1])+(1-conf_knn_lasso_full$byClass[2]))
AM_knn<- c(AM_knn_full,AM_knn_logistic,AM_knn_lasso, AM_knn_logistic_full, AM_knn_lasso_full)
names(AM_knn)<-  c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
round(AM_knn,3)
##
#Missclassification  rate------------------------------------------
MC_knn_full<-1 - conf_knn_full$overall[1]
MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
MC_knn_lasso<-1- conf_knn_lasso$overall[1]
MC_knn_logistic_full<-1 - conf_knn_logistic_full$overall[1]
MC_knn_lasso_full<-1- conf_knn_lasso_full$overall[1]
MC_knn<- c(MC_knn_full,MC_knn_logistic,MC_knn_lasso,MC_knn_logistic_full,MC_knn_lasso_full)
names(MC_knn)<-  c("Full data model", "DR via Logistic","DR via Lasso","DR via Logistic Full","DR via Lasso Full")
round(MC_knn,3)



######################################
###ROC CURVE

#KNN------------------------------------------
pred_knn_full <- predict(knn_full, newdata = test_data[,-ncol(test_data)], type = "prob")[, 2]
pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
#
pred_knn_logistic_full <- predict(knn_logistic_full, newdata = x_test_transformed_logistic_full, type = "prob")[, 2]
pred_knn_lasso_full <- predict(knn_lasso_full, newdata = x_test_transformed_lasso_full, type = "prob")[, 2]

#KNN----------------------------
pred_knn_full <- prediction(pred_knn_full, test_data[,ncol(test_data)])
pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
#
pred_knn_logistic_full <- prediction(pred_knn_logistic_full, test_data[,ncol(test_data)])
pred_knn_lasso_full <- prediction(pred_knn_lasso_full, test_data[,ncol(test_data)])
# pred_knn_full <- prediction(attr(knn_full_prob, "prob"), test_data[,ncol(test_data)])
# pred_knn_logistic <- prediction(attr(knn_logistic_prob, "prob"), test_data[,ncol(test_data)])
# pred_knn_lasso <- prediction(attr(knn_lasso_prob, "prob"), test_data[,ncol(test_data)])
# #
# pred_knn_logistic_full <- prediction(attr(knn_logistic_full_prob, "prob"), test_data[,ncol(test_data)])
# pred_knn_lasso_full <- prediction(attr(knn_lasso_full_prob, "prob"), test_data[,ncol(test_data)])
#
#perf_logis  <- performance(pred_logis, "tpr", "fpr" )


#KNN--------------------------
perf_knn_full  <- performance(pred_knn_full, "tpr", "fpr")
perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
#
perf_knn_logistic_full  <- performance(pred_knn_logistic_full, "tpr", "fpr")
perf_knn_lasso_full  <- performance(pred_knn_lasso_full, "tpr", "fpr")

#Knn----
plot(perf_knn_full,colorize = FALSE, col="black", main="", lty=1, lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
plot(perf_knn_full,colorize = FALSE, add=TRUE,col="black", main="ROC curves for models fitted through knn", lty=1, lwd=3)
plot(perf_knn_logistic,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=3)
plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="blue", lty=1,lwd=3)
plot(perf_knn_logistic_full,add=TRUE, colorize = FALSE, col="skyblue",lty=1,lwd=3)
plot(perf_knn_lasso_full,add=TRUE, colorize = FALSE, col="green", lty=1,lwd=3)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
legend("bottomright",legend = c("Full model", expression(paste("LLO(", lambda, "= 0)")), 
                                expression(paste("LLO(", lambda, "> 0)")) ),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=3)




#Area under the curve for knn------------------------------------------------------------------
AUC_knn<-c(Auc_full=performance(pred_knn_full, "auc")@y.values[[1]], AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
           AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_logistic_full=performance(pred_knn_logistic_full, "auc")@y.values[[1]],
           AUC_lasso_full=performance(pred_knn_lasso_full, "auc")@y.values[[1]])

AUC_knn

#-------------------------------------------------------------------------------------------------------------------
# paste(round(c(F1_rf[1], accuracy_rf[1],rf_time.taken[1], AM_rf[1], F1_knn[1], accuracy_knn[1],knn_time.taken[1], AM_knn[1]),3), collapse = " & ")
# paste(round(c(F1_rf[2], accuracy_rf[2],rf_time.taken[2], AM_rf[2], F1_knn[2], accuracy_knn[2],knn_time.taken[2], AM_knn[2]),3), collapse = " & ")
#   paste(round(c(F1_rf[3], accuracy_rf[3],rf_time.taken[3], AM_rf[3], F1_knn[3], accuracy_knn[3],knn_time.taken[3], AM_knn[3]),3), collapse = " & ")
# paste(round(c(F1_rf[4], accuracy_rf[4],rf_time.taken[4], AM_rf[4], F1_knn[4], accuracy_knn[4],knn_time.taken[4], AM_knn[4]),3), collapse = " & ")
# paste(round(c(F1_rf[5], accuracy_rf[5],rf_time.taken[5], AM_rf[5], F1_knn[5], accuracy_knn[5],knn_time.taken[5], AM_knn[5]),3), collapse = " & ")
# 


paste(round(c( AM_knn[1], MC_knn[1], AUC_knn[1],knn_time.taken[1]),3), collapse = " & ")
paste(round(c( AM_knn[2], MC_knn[2], AUC_knn[2],knn_time.taken[2]),3), collapse = " & ")
paste(round(c( AM_knn[3], MC_knn[3], AUC_knn[3],knn_time.taken[3]),3), collapse = " & ")
paste(round(c( AM_knn[4], MC_knn[4], AUC_knn[4],knn_time.taken[4]),3), collapse = " & ")
paste(round(c( AM_knn[5], MC_knn[5], AUC_knn[5],knn_time.taken[5]),3), collapse = " & ")



#Dimention selection throug RF----------------------------
dimensions_logistic<-ncomp_selection3(traindata=train_transformed_logistic_full, testdata=test_transformed_logistic_full, method=c("rf"),cv=TRUE)
dimensions_lasso<-ncomp_selection3(traindata=train_transformed_lasso_full, testdata=test_transformed_lasso_full, method=c("rf"),cv=TRUE)
plot(dimensions_logistic, type="l",ylim=c(0.7,1) ,ylab="Accuracy", xlab="Dimensions", main="Dimension selection for HV data", lwd=2 )
#For grid---------------------
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")
# Add white grid
grid(nx = NULL, ny = NULL,
     col = "white", lwd = 1)

lines(dimensions_logistic, type="l", lwd=2 , col="black")
desired_components_logistic <- which.max(dimensions_logistic)[1]
abline(v=desired_components_logistic, lty=2, col="black")
#lasso
lines(dimensions_lasso, type="l" , lwd=2, col="red")
desired_components_lasso <- which.max(dimensions_lasso)[1]
abline(v=desired_components_lasso, lty=2, col="red")
legend("topleft", 
       legend = c(expression(paste("LLO(", lambda, "= 0)")), 
                  expression(paste("LLO(", lambda, "> 0)"))),
       col = c("black", "red"), lty = 1, lwd = 2,  bty='n', inset=c(0.65, 0.01))




#k <-desired_components  # Number of selected components
Vk_logistic <- svd_logistic$v[, 1:desired_components_logistic]
Vk_lasso <- svd_lasso$v[, 1:desired_components_lasso]
######Transform for logistic and Lasso 
x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
colnames(x_train_transformed_logistic) <- paste0("PC", 1:desired_components_logistic)
colnames(x_train_transformed_lasso) <- paste0("PC", 1:desired_components_lasso)
colnames(x_test_transformed_logistic) <- paste0("PC", 1:desired_components_logistic)
colnames(x_test_transformed_lasso) <- paste0("PC", 1:desired_components_lasso)
# mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])



# Random Forest model-------------------------------------------------------
mtry_tune <- tuneRF(train_data[,-ncol(train_data)], train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_full<- randomForest(x = train_data[,-ncol(train_data)],y =train_data$y,    mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_full <- round(end.time - start.time,2)

#logistic----------------------------------------------
mtry_tune <- tuneRF(x_train_transformed_logistic, train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_logistic <-  randomForest(x = x_train_transformed_logistic,y =train_data$y,   mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_logistic <- round(end.time - start.time,2)
#lasso------------------------------------------------
mtry_tune <- tuneRF(x_train_transformed_lasso, train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_lasso <-randomForest(x = x_train_transformed_lasso,y =train_data$y,   mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_lasso <- round(end.time - start.time,2)
#
#logistic fully projected space-----------------------
mtry_tune <- tuneRF(x_train_transformed_logistic_full, train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_logistic_full <- randomForest(x = x_train_transformed_logistic_full,y =train_data$y,    mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_logistic_full <- round(end.time - start.time,2)
##lasso fully projected space-----------------------
mtry_tune <- tuneRF(x_train_transformed_lasso_full, train_data$y, ntreeTry = 500, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = FALSE)
best_mtry <- mtry_tune[mtry_tune[, 2] == min(mtry_tune[, 2]), 1]
best_mtry<- min(best_mtry)
#
start.time <- Sys.time()
rf_lasso_full <- randomForest(x = x_train_transformed_lasso_full,y =train_data$y,   mtry = best_mtry, ntree = ntree, importance = TRUE, proximity = TRUE)
end.time <- Sys.time()
time.taken_lasso_full <- round(end.time - start.time,2)
#
rf_time.taken<- c(time.taken_full,time.taken_logistic,time.taken_lasso,time.taken_logistic_full,time.taken_lasso_full)
names(rf_time.taken)<- c("full data model", "logistic", "lasso", "logistic_full", "lasso_full")
print(rf_time.taken)

#plot(rf_model$err.rate, lwd="2")
min_error_tree<- c(which.min(rf_full$err.rate[,1]),which.min(rf_logistic$err.rate[,1]),which.min(rf_lasso$err.rate[,1]),
                   which.min(rf_logistic_full$err.rate[,1]),which.min(rf_lasso_full$err.rate[,1]))
names(min_error_tree)<- c("full data model", "logistic", "lasso", "logistic_full", "lasso_full")
print(min_error_tree)
# plot(rf_full)
# plot(rf_logistic, add=TRUE)
# plot(rf_lasso, add=TRUE)
# rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
# legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)

plot(rf_full$err.rate[,1],col="black", type="l", ylim=c(0.0,1), ylab="Error",xlab="Trees", main="OBB Error", lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
lines(rf_full$err.rate[,1],col="black", type="l", ylim=c(0.0,1), ylab="Error",xlab="Trees", main="OBB Error", lwd=3)
lines(rf_logistic$err.rate[,1],col="red", type="l",lwd=3)
lines(rf_lasso$err.rate[,1],col="blue", type="l",lwd=3)
lines(rf_logistic_full$err.rate[,1],col="skyblue", type="l",lwd=3)
lines(rf_lasso_full$err.rate[,1],col="green", type="l",lwd=3)
abline(v=which.min(rf_full$err.rate[,1]), lty=2)
abline(v=which.min(rf_logistic$err.rate[,1]), lty=2, col="red")
abline(v=which.min(rf_lasso$err.rate[,1]), lty=2, col="blue")
abline(v=which.min(rf_logistic_full$err.rate[,1]), lty=2, col="skyblue")
abline(v=which.min(rf_lasso_full$err.rate[,1]), lty=2, col="green")
#legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso"),col = c("black", "red","blue"), lty = 1)
legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso"),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=3)

#
plot(rf_full$err.rate[,2],col="black", type="l", ylim=c(0.0,0.5), ylab="Error",xlab="Trees", main="Error for class 0",lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
lines(rf_full$err.rate[,2],col="black", type="l", ylim=c(0.0,0.5), ylab="Error",xlab="Trees", main="Error for class 0",lwd=3)
lines(rf_logistic$err.rate[,2],col="red", type="l",lwd=3)
lines(rf_lasso$err.rate[,2],col="blue", type="l",lwd=3)
#
lines(rf_logistic_full$err.rate[,2],col="skyblue", type="l",lwd=3)
lines(rf_lasso_full$err.rate[,2],col="green", type="l",lwd=3)

abline(v=which.min(rf_full$err.rate[,2]), lty=2)
abline(v=which.min(rf_logistic$err.rate[,2]), lty=2, col="red")
abline(v=which.min(rf_lasso$err.rate[,2]), lty=2, col="blue")
abline(v=which.min(rf_logistic_full$err.rate[,2]), lty=2, col="skyblue")
abline(v=which.min(rf_lasso_full$err.rate[,2]), lty=2, col="green")
#legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso"),col = c("black", "red","blue"), lty = 1)
legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso"),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=3)

#
plot(rf_full$err.rate[,3],col="black", type="l", ylim=c(0.2,1.7), ylab="Error",xlab="Trees", main="Error for class 1", lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
lines(rf_full$err.rate[,3],col="black", type="l", ylim=c(0.2,1.7), ylab="Error",xlab="Trees", main="Error for class 1", lwd=3)
lines(rf_logistic$err.rate[,3],col="red", type="l", lwd=3)
lines(rf_lasso$err.rate[,3],col="blue", type="l", lwd=3)
#
lines(rf_logistic_full$err.rate[,3],col="skyblue", type="l", lwd=3)
lines(rf_lasso_full$err.rate[,3],col="green", type="l", lwd=3)

abline(v=which.min(rf_full$err.rate[,3]), lty=2, col="black")
abline(v=which.min(rf_logistic$err.rate[,3]), lty=2, col="red")
abline(v=which.min(rf_lasso$err.rate[,3]), lty=2, col="blue")
abline(v=which.min(rf_logistic_full$err.rate[,3]), lty=2, col="skyblue")
abline(v=which.min(rf_lasso_full$err.rate[,3]), lty=2, col="green")
#legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso"),col = c("black", "red","blue"), lty = 1)
legend("topright", legend = c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso"),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1, lwd=3)

#importance(rf_model)
#varImpPlot(rf_model)
## Look at variable importance:
#round(importance(rf_model), 1)

#########_--------------------------------------------------------------------------------
# Predict the response using the transformed test data
predict_rf_full <- predict(rf_full, newdata= test_data[,-ncol(test_data)], type="response")
predict_rf_logistic <- predict(rf_logistic, newdata= x_test_transformed_logistic, type="response",norm.votes=TRUE)
predict_rf_lasso <- predict(rf_lasso, newdata= x_test_transformed_lasso, type="response",norm.votes=TRUE)
#
predict_rf_logistic_full <- predict(rf_logistic_full, newdata= x_test_transformed_logistic_full, type="response",norm.votes=TRUE)
predict_rf_lasso_full <- predict(rf_lasso_full, newdata= x_test_transformed_lasso_full, type="response",norm.votes=TRUE)

#Random forest---------
conf_rf_full<-confusionMatrix(as.factor(predict_rf_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_rf_logistic<-confusionMatrix(as.factor(predict_rf_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_rf_lasso<-confusionMatrix(as.factor(predict_rf_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")

conf_rf_logistic_full<-confusionMatrix(as.factor(predict_rf_logistic_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_rf_lasso_full<-confusionMatrix(as.factor(predict_rf_lasso_full),as.factor(test_data[,ncol(test_data)]),mode = "everything")

# F1_rf<- c(conf_rf_full$byClass[7],conf_rf_logistic$byClass[7],conf_rf_lasso$byClass[7],conf_rf_logistic_full$byClass[7],conf_rf_lasso_full$byClass[7])
# names(F1_rf)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(F1_rf,3)

# 
# accuracy_rf<- c(conf_rf_full$byClass[11],conf_rf_logistic$byClass[11],conf_rf_lasso$byClass[11],conf_rf_logistic_full$byClass[11],conf_rf_lasso_full$byClass[11])
# names(accuracy_rf)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(accuracy_rf,3)
#AM risk------------------------------------------#AM risk------------------------------------------
AM_rf_full<-(1/2)*((1-conf_rf_full$byClass[1])+(1-conf_rf_full$byClass[2]))
AM_rf_logistic<-(1/2)*((1-conf_rf_logistic$byClass[1])+(1-conf_rf_logistic$byClass[2]))
AM_rf_lasso<-(1/2)*((1-conf_rf_lasso$byClass[1])+(1-conf_rf_lasso$byClass[2]))
AM_rf_logistic_full<- (1/2)*((1-conf_rf_logistic_full$byClass[1])+(1-conf_rf_logistic_full$byClass[2]))
AM_rf_lasso_full<- (1/2)*((1-conf_rf_lasso_full$byClass[1])+(1-conf_rf_lasso_full$byClass[2]))
AM_rf<- c(AM_rf_full,AM_rf_logistic,AM_rf_lasso, AM_rf_logistic_full, AM_rf_lasso_full)
names(AM_rf)<-  c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
round(AM_rf,3)
#Missclassification  rate------------------------------------------
MC_rf_full<-1 - conf_rf_full$overall[1]
MC_rf_logistic<-1 - conf_rf_logistic$overall[1]
MC_rf_lasso<-1- conf_rf_lasso$overall[1]
MC_rf_logistic_full<-1 - conf_rf_logistic_full$overall[1]
MC_rf_lasso_full<-1- conf_rf_lasso_full$overall[1]
MC_rf<- c(MC_rf_full,MC_rf_logistic,MC_rf_lasso,MC_rf_logistic_full,MC_rf_lasso_full)
names(MC_rf)<-  c("Full data model", "DR via Logistic","DR via Lasso","DR via Logistic Full","DR via Lasso Full")
round(MC_rf,3)

######################################
###ROC CURVE
# Predict probabilities for the test set (Random forest)----------------------------------
pred_rf_full <- predict(rf_full, newdata = test_data[,-ncol(test_data)], type = "prob")[, 2]
pred_rf_logistic <- predict(rf_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
pred_rf_lasso <- predict(rf_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
#
pred_rf_logistic_full <- predict(rf_logistic_full, newdata = x_test_transformed_logistic_full, type = "prob")[, 2]
pred_rf_lasso_full <- predict(rf_lasso_full, newdata = x_test_transformed_lasso_full, type = "prob")[, 2]
#

# Compute the ROC curve for logistic regression
pred_rf_full <- prediction(pred_rf_full, test_data[,ncol(test_data)])
pred_rf_logistic <- prediction(pred_rf_logistic, test_data[,ncol(test_data)])
pred_rf_lasso <- prediction(pred_rf_lasso, test_data[,ncol(test_data)])
#
pred_rf_logistic_full <- prediction(pred_rf_logistic_full, test_data[,ncol(test_data)])
pred_rf_lasso_full <- prediction(pred_rf_lasso_full, test_data[,ncol(test_data)])

#perf_logis  <- performance(pred_logis, "tpr", "fpr" )
perf_rf_full  <- performance(pred_rf_full, "tpr", "fpr")
perf_rf_logistic  <- performance(pred_rf_logistic, "tpr", "fpr")
perf_rf_lasso  <- performance(pred_rf_lasso, "tpr", "fpr")
#
perf_rf_logistic_full  <- performance(pred_rf_logistic_full, "tpr", "fpr")
perf_rf_lasso_full  <- performance(pred_rf_lasso_full, "tpr", "fpr")


#plot( perf_logis, colorize = FALSE,main = "ROC Curve")

plot(perf_rf_full, colorize = FALSE, main="",lwd=3)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
plot(perf_rf_full, colorize = FALSE,add=TRUE, main="ROC curve for models fitted through RF",lwd=3)

plot(perf_rf_logistic,add=TRUE, colorize = FALSE, col="red",lwd=3)
plot(perf_rf_lasso,add=TRUE, colorize = FALSE, col="blue",lwd=3)
plot(perf_rf_logistic_full,add=TRUE, colorize = FALSE, col="skyblue",lwd=3)
plot(perf_rf_lasso_full,add=TRUE, colorize = FALSE, col="green",lwd=3)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

legend("bottomright",legend = c("Full model", expression(paste("LLO(", lambda, "= 0)")), 
                                expression(paste("LLO(", lambda, "> 0)")) ),
       col = c("black", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=3)


#Area under the curve Random forest------------------------------------------------------------------------

AUC_rf<-c(Auc_full=performance(pred_rf_full, "auc")@y.values[[1]], AUC_logsitc=performance(pred_rf_logistic, "auc")@y.values[[1]],
          AUC_lasso=performance(pred_rf_lasso, "auc")@y.values[[1]], AUC_logistic_full=performance(pred_rf_logistic_full, "auc")@y.values[[1]],
          AUC_lasso_full=performance(pred_rf_lasso_full, "auc")@y.values[[1]])

AUC_rf

#-------------------------------------------------------------------------------------------------------------------
# paste(round(c(F1_rf[1], accuracy_rf[1],rf_time.taken[1], AM_rf[1], F1_knn[1], accuracy_knn[1],knn_time.taken[1], AM_knn[1]),3), collapse = " & ")
# paste(round(c(F1_rf[2], accuracy_rf[2],rf_time.taken[2], AM_rf[2], F1_knn[2], accuracy_knn[2],knn_time.taken[2], AM_knn[2]),3), collapse = " & ")
#   paste(round(c(F1_rf[3], accuracy_rf[3],rf_time.taken[3], AM_rf[3], F1_knn[3], accuracy_knn[3],knn_time.taken[3], AM_knn[3]),3), collapse = " & ")
# paste(round(c(F1_rf[4], accuracy_rf[4],rf_time.taken[4], AM_rf[4], F1_knn[4], accuracy_knn[4],knn_time.taken[4], AM_knn[4]),3), collapse = " & ")
# paste(round(c(F1_rf[5], accuracy_rf[5],rf_time.taken[5], AM_rf[5], F1_knn[5], accuracy_knn[5],knn_time.taken[5], AM_knn[5]),3), collapse = " & ")
# 


paste(round(c(AM_rf[1], MC_rf[1],AUC_rf[1],rf_time.taken[1]),3), collapse = " & ") 
paste(round(c(AM_rf[2], MC_rf[2],AUC_rf[2],rf_time.taken[2]), 3), collapse = " & ")
paste(round(c(AM_rf[3], MC_rf[3],AUC_rf[3],rf_time.taken[3]), 3), collapse = " & ")
paste(round(c(AM_rf[4], MC_rf[4],AUC_rf[4],rf_time.taken[4]), 3), collapse = " & ")
paste(round(c(AM_rf[5], MC_rf[5],AUC_rf[5],rf_time.taken[5]),3), collapse = " & ")



