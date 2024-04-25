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
#
train_test_split = function(X, y, test_size, seed){
  set.seed(seed)
  n=nrow(X)
  test_id = sample(n, round(n*test_size))
  list_final = list("X_train" = X[-test_id,], "X_test" = X[test_id,], 
                    "y_train" = y[-test_id], "y_test" = y[test_id])
  return(list_final)
}
par(mar = c(4, 4, 2, 0.5)) 
par(mfrow=c(2,3))
#
data<- read.csv("hill-valley_csv.csv")
table(data$Class)
data$y<-data$Class
data <- subset(data, select = -Class)
#data<- scale(data[-ncol(data)])
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

svd_logistic <- svd(coef.mat_logistic)
svd_lasso <- svd(coef.mat_lasso)
##
#compititors
save.fit =dr(train_data$y~.,data=train_data[,-ncol(train_data)], method="save")
phd.fit = dr(as.numeric(train_data$y)~.,train_data[,-ncol(train_data)], method="phdy")
potd.fit<-potd(X=as.matrix(train_data[,-ncol(train_data)]), y=train_data$y, ndim=ncol(train_data[,-ncol(train_data)]))

###############



d=2


Vk_logistic <- svd_logistic$v[, 1:d]
Vk_lasso <- svd_lasso$v[,  1:d]
Vk_save <- save.fit$evectors[,  1:d]
Vk_phd <- phd.fit$evectors[,  1:d]
Vk_potd <- potd.fit[,  1:d]
#data for prediction-----------------------------------------------------


######Transform for logistic and Lasso 
x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_save
x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_phd
x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
colnames(x_train_transformed_save) <- paste0("PC", 1:d)
colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
colnames(x_train_transformed_potd) <- paste0("PC", 1:d)

x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_save
x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_phd
x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
colnames(x_test_transformed_save) <- paste0("PC", 1:d)
colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
# mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])

###KNN-------------------------------------------------------------------
k_range <- 10 # Example range of k values: 1, 3, 5, 7, 9

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
knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_save<- round(end.time - start.time,2)
#knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)


#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_phd <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)

#
#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_potd <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)



knn_time_d2<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
names(knn_time_d2)<- c("logistic","lasso", "save","phd","potd")
print(knn_time_d2)


# plot(rf_full)
# plot(rf_logistic, add=TRUE)
# plot(rf_lasso, add=TRUE)
# rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
# legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)

#KNNprediction------------------------------------------------------------------------------

predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
#Confusion matrix  (KNN)------------------------------------
conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
#F1 score-----------------------
# F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
# names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(F1_knn,3)
#accuracy_knn-----------------------------
# accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
# names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(accuracy_knn,3)
#AM risk------------------------------------------
AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
AM_d2<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
names(AM_d2)<-  c("logistic","lasso", "save", "phd", "potd")
round(AM_d2,3)
##
#Missclassification  rate------------------------------------------

MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
MC_knn_lasso<-1- conf_knn_lasso$overall[1]
MC_knn_save<-1 - conf_knn_save$overall[1] 
MC_knn_phd<-1 - conf_knn_phd$overall[1] 
MC_knn_potd<-1 - conf_knn_potd$overall[1] 
MC_d2<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
names(MC_d2)<-  c("logistic","lasso", "save", "phd", "potd")
round(MC_d2,3)



F1_d2<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
names(F1_d2)<- c( "logistic","lasso", "save","phd","potd")



######################################
###ROC CURVE------------------------------------------------------------------

#KNN------------------------------------------
pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
# Prediction for the ROC------------------------------
#KNN----------------------------
pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
#Performance----------------------------

#KNN--------------------------
perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")
#Plot------------------------------------


#ROC
plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 2")), lty=1, lwd=2)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
# legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
#                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
#                                 "POTD" ),
#        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)




#Knn----
# plot(perf_knn_logistic,colorize = FALSE, col="black", main="ROC curves for knn fits", lty=1, lwd=2)
# rect(par("usr")[1], par("usr")[3],
#      par("usr")[2], par("usr")[4],
#      col = "#ebebeb")
# 
# # Add white grid
# grid(nx = NULL, ny = NULL,
#      col = "gray", lwd = 1)
# plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="black", main="ROC curves for models fitted through knn", lty=1, lwd=2)
# plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="purple",lty=1,lwd=2)
# plot(perf_knn_save,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=2)
#plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=2)
#plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=2)
# abline(a=0,b=1,lwd=2,lty=2,col="gray")
# legend("bottomright",legend = c( expression(paste("LLO(", lambda, "= 0)")), 
#                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE" ,"pHd" ,"potd" ),
#        col = c("black","purple", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=2)


#Area under the curve for knn------------------------------------------------------------------
AUC_d2<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
           AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
           ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])

AUC_d2




###############



d=3


Vk_logistic <- svd_logistic$v[, 1:d]
Vk_lasso <- svd_lasso$v[,  1:d]
Vk_save <- save.fit$evectors[,  1:d]
Vk_phd <- phd.fit$evectors[,  1:d]
Vk_potd <- potd.fit[,  1:d]
#data for prediction-----------------------------------------------------


######Transform for logistic and Lasso 
x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_save
x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_phd
x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
colnames(x_train_transformed_save) <- paste0("PC", 1:d)
colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
colnames(x_train_transformed_potd) <- paste0("PC", 1:d)

x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_save
x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_phd
x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
colnames(x_test_transformed_save) <- paste0("PC", 1:d)
colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
# mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])


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
knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_save<- round(end.time - start.time,2)
#knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)


#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_phd <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)

#
#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_potd <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)



knn_time_d3<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
names(knn_time_d3)<- c("logistic","lasso", "save","phd","potd")
print(knn_time_d3)

# plot(rf_full)
# plot(rf_logistic, add=TRUE)
# plot(rf_lasso, add=TRUE)
# rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
# legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)

#KNNprediction------------------------------------------------------------------------------

predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
#Confusion matrix  (KNN)------------------------------------
conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
#F1 score-----------------------
# F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
# names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(F1_knn,3)
#accuracy_knn-----------------------------
# accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
# names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(accuracy_knn,3)
#AM risk------------------------------------------
AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
AM_d3<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
names(AM_d3)<-  c("logistic","lasso", "save", "phd", "potd")
round(AM_d3,3)
##
#Missclassification  rate------------------------------------------

MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
MC_knn_lasso<-1- conf_knn_lasso$overall[1]
MC_knn_save<-1 - conf_knn_save$overall[1] 
MC_knn_phd<-1 - conf_knn_phd$overall[1] 
MC_knn_potd<-1 - conf_knn_potd$overall[1] 
MC_d3<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
names(MC_d3)<-  c("logistic","lasso", "save", "phd", "potd")
round(MC_d3,3)




F1_d3<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
names(F1_d3)<- c( "logistic","lasso", "save","phd","potd")

######################################
###ROC CURVE------------------------------------------------------------------

#KNN------------------------------------------
pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
# Prediction for the ROC------------------------------
#KNN----------------------------
pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
#Performance----------------------------

#KNN--------------------------
perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")
#Plot------------------------------------
#ROC
plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 3")), lty=1, lwd=2)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
# legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
#                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
#                                 "POTD" ),
#        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)
# 

#Knn----
# plot(perf_knn_logistic,colorize = FALSE, col="black", main="ROC curves for knn fits", lty=1, lwd=2)
# rect(par("usr")[1], par("usr")[3],
#      par("usr")[2], par("usr")[4],
#      col = "#ebebeb")
# 
# # Add white grid
# grid(nx = NULL, ny = NULL,
#      col = "gray", lwd = 1)
# plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="black", main="ROC curves for models fitted through knn", lty=1, lwd=2)
# plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="purple",lty=1,lwd=2)
# plot(perf_knn_save,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=2)
#plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=2)
#plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=2)
# abline(a=0,b=1,lwd=2,lty=2,col="gray")
# legend("bottomright",legend = c( expression(paste("LLO(", lambda, "= 0)")), 
#                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE" ,"pHd" ,"potd" ),
#        col = c("black","purple", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=2)


#Area under the curve for knn------------------------------------------------------------------
AUC_d3<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
           AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
           ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])

AUC_d3



###############



d=4


Vk_logistic <- svd_logistic$v[, 1:d]
Vk_lasso <- svd_lasso$v[,  1:d]
Vk_save <- save.fit$evectors[,  1:d]
Vk_phd <- phd.fit$evectors[,  1:d]
Vk_potd <- potd.fit[,  1:d]
#data for prediction-----------------------------------------------------


######Transform for logistic and Lasso 
x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_save
x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_phd
x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
colnames(x_train_transformed_save) <- paste0("PC", 1:d)
colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
colnames(x_train_transformed_potd) <- paste0("PC", 1:d)

x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_save
x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_phd
x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
colnames(x_test_transformed_save) <- paste0("PC", 1:d)
colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
# mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])

###KNN-------------------------------------------------------------------
k_range <- 10 # Example range of k values: 1, 3, 5, 7, 9

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
knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_save<- round(end.time - start.time,2)
#knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)


#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_phd <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)

#
#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_potd <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)



knn_time_d4<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
names(knn_time_d4)<- c("logistic","lasso", "save","phd","potd")
print(knn_time_d4)

# plot(rf_full)
# plot(rf_logistic, add=TRUE)
# plot(rf_lasso, add=TRUE)
# rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
# legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)

#KNNprediction------------------------------------------------------------------------------

predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
#Confusion matrix  (KNN)------------------------------------
conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
#F1 score-----------------------
# F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
# names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(F1_knn,3)
#accuracy_knn-----------------------------
# accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
# names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(accuracy_knn,3)
#AM risk------------------------------------------
AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
AM_d4<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
names(AM_d4)<-  c("logistic","lasso", "save", "phd", "potd")
round(AM_d4,3)
##
#Missclassification  rate------------------------------------------

MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
MC_knn_lasso<-1- conf_knn_lasso$overall[1]
MC_knn_save<-1 - conf_knn_save$overall[1] 
MC_knn_phd<-1 - conf_knn_phd$overall[1] 
MC_knn_potd<-1 - conf_knn_potd$overall[1] 
MC_d4<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
names(MC_d4)<-  c("logistic","lasso", "save", "phd", "potd")
round(MC_d4,3)




F1_d4<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
names(F1_d4)<- c( "logistic","lasso", "save","phd","potd")

######################################
###ROC CURVE------------------------------------------------------------------

#KNN------------------------------------------
pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
# Prediction for the ROC------------------------------
#KNN----------------------------
pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
#Performance----------------------------

#KNN--------------------------
perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")
#Plot------------------------------------

#ROC
plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 4")), lty=1, lwd=2)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
# legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
#                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
#                                 "POTD" ),
#        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)
#Knn----
# plot(perf_knn_logistic,colorize = FALSE, col="black", main="ROC curves for knn fits", lty=1, lwd=2)
# rect(par("usr")[1], par("usr")[3],
#      par("usr")[2], par("usr")[4],
#      col = "#ebebeb")
# 
# # Add white grid
# grid(nx = NULL, ny = NULL,
#      col = "gray", lwd = 1)
# plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="black", main="ROC curves for models fitted through knn", lty=1, lwd=2)
# plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="purple",lty=1,lwd=2)
# plot(perf_knn_save,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=2)
#plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=2)
#plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=2)
# abline(a=0,b=1,lwd=2,lty=2,col="gray")
# legend("bottomright",legend = c( expression(paste("LLO(", lambda, "= 0)")), 
#                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE" ,"pHd" ,"potd" ),
#        col = c("black","purple", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=2)


#Area under the curve for knn------------------------------------------------------------------
AUC_d4<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
           AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
           ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])

AUC_d4



###############



d=5


Vk_logistic <- svd_logistic$v[, 1:d]
Vk_lasso <- svd_lasso$v[,  1:d]
Vk_save <- save.fit$evectors[,  1:d]
Vk_phd <- phd.fit$evectors[,  1:d]
Vk_potd <- potd.fit[,  1:d]
#data for prediction-----------------------------------------------------


######Transform for logistic and Lasso 
x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_save
x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_phd
x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
colnames(x_train_transformed_save) <- paste0("PC", 1:d)
colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
colnames(x_train_transformed_potd) <- paste0("PC", 1:d)

x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_save
x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_phd
x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
colnames(x_test_transformed_save) <- paste0("PC", 1:d)
colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
# mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])

###KNN-------------------------------------------------------------------

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
knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_save<- round(end.time - start.time,2)
#knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)


#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_phd <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)

#
#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_potd <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)



knn_time_d5<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
names(knn_time_d5)<- c("logistic","lasso", "save","phd","potd")
print(knn_time_d5)

# plot(rf_full)
# plot(rf_logistic, add=TRUE)
# plot(rf_lasso, add=TRUE)
# rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
# legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)

#KNNprediction------------------------------------------------------------------------------

predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
#Confusion matrix  (KNN)------------------------------------
conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
#F1 score-----------------------
# F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
# names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(F1_knn,3)
#accuracy_knn-----------------------------
# accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
# names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(accuracy_knn,3)
#AM risk------------------------------------------
AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
AM_d5<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
names(AM_d5)<-  c("logistic","lasso", "save", "phd", "potd")
round(AM_d5,3)
##
#Missclassification  rate------------------------------------------

MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
MC_knn_lasso<-1- conf_knn_lasso$overall[1]
MC_knn_save<-1 - conf_knn_save$overall[1] 
MC_knn_phd<-1 - conf_knn_phd$overall[1] 
MC_knn_potd<-1 - conf_knn_potd$overall[1] 
MC_d5<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
names(MC_d5)<-  c("logistic","lasso", "save", "phd", "potd")
round(MC_d5,3)



F1_d5<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
names(F1_d5)<- c( "logistic","lasso", "save","phd","potd")

######################################
###ROC CURVE------------------------------------------------------------------

#KNN------------------------------------------
pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
# Prediction for the ROC------------------------------
#KNN----------------------------
pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
#Performance----------------------------

#KNN--------------------------
perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")
#Plot------------------------------------
#ROC
plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 5")), lty=1, lwd=2)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
# legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
#                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
#                                 "POTD" ),
#        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)
#Knn----
# plot(perf_knn_logistic,colorize = FALSE, col="black", main="ROC curves for knn fits", lty=1, lwd=2)
# rect(par("usr")[1], par("usr")[3],
#      par("usr")[2], par("usr")[4],
#      col = "#ebebeb")
# 
# # Add white grid
# grid(nx = NULL, ny = NULL,
#      col = "gray", lwd = 1)
# plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="black", main="ROC curves for models fitted through knn", lty=1, lwd=2)
# plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="purple",lty=1,lwd=2)
# plot(perf_knn_save,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=2)
#plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=2)
#plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="red",lty=1,lwd=2)
# abline(a=0,b=1,lwd=2,lty=2,col="gray")
# legend("bottomright",legend = c( expression(paste("LLO(", lambda, "= 0)")), 
#                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE" ,"pHd" ,"potd" ),
#        col = c("black","purple", "red","blue","skyblue","green", "olivedrab","brown"), lty = 1,lwd=2)


#Area under the curve for knn------------------------------------------------------------------
AUC_d5<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
           AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
           ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])

AUC_d5



###############



d=10


Vk_logistic <- svd_logistic$v[, 1:d]
Vk_lasso <- svd_lasso$v[,  1:d]
Vk_save <- save.fit$evectors[,  1:d]
Vk_phd <- phd.fit$evectors[,  1:d]
Vk_potd <- potd.fit[,  1:d]
#data for prediction-----------------------------------------------------


######Transform for logistic and Lasso 
x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_save
x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_phd
x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
colnames(x_train_transformed_save) <- paste0("PC", 1:d)
colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
colnames(x_train_transformed_potd) <- paste0("PC", 1:d)

x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_save
x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_phd
x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
colnames(x_test_transformed_save) <- paste0("PC", 1:d)
colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
# mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])

###KNN-------------------------------------------------------------------

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
knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_save<- round(end.time - start.time,2)
#knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)


#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_phd <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)

#
#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_potd <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)



knn_time_d10<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
names(knn_time_d10)<- c("logistic","lasso", "save","phd","potd")
print(knn_time_d10)

# plot(rf_full)
# plot(rf_logistic, add=TRUE)
# plot(rf_lasso, add=TRUE)
# rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
# legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)

#KNNprediction------------------------------------------------------------------------------

predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
#Confusion matrix  (KNN)------------------------------------
conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
#F1 score-----------------------
# F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
# names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(F1_knn,3)
#accuracy_knn-----------------------------
# accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
# names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(accuracy_knn,3)
#AM risk------------------------------------------
AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
AM_d10<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
names(AM_d10)<-  c("logistic","lasso", "save", "phd", "potd")
round(AM_d10,3)
##
#Missclassification  rate------------------------------------------

MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
MC_knn_lasso<-1- conf_knn_lasso$overall[1]
MC_knn_save<-1 - conf_knn_save$overall[1] 
MC_knn_phd<-1 - conf_knn_phd$overall[1] 
MC_knn_potd<-1 - conf_knn_potd$overall[1] 
MC_d10<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
names(MC_d10)<-  c("logistic","lasso", "save", "phd", "potd")
round(MC_d10,3)


F1_d10<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
names(F1_d10)<- c( "logistic","lasso", "save","phd","potd")


######################################
###ROC CURVE------------------------------------------------------------------

#KNN------------------------------------------
pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
# Prediction for the ROC------------------------------
#KNN----------------------------
pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
#Performance----------------------------

#KNN--------------------------
perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")
#Plot------------------------------------
#ROC
plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 10")), lty=1, lwd=2)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
# legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
#                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
#                                 "POTD" ),
#        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)
# Create an empty plot
plot(0, type = "n", xlim = c(0, 1), ylim = c(0, 1), xlab = "", ylab = "", main=expression(paste(  "Legends")))
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
# Add legends
legend("center",
       legend = c(expression(paste("LLO(", lambda, "= 0)")),
                  expression(paste("LLO(", lambda, "> 0)")),
                  "SAVE", "PHD", "POTD"),
       col = c("darkorange", "olivedrab", "springgreen", "cyan2", "darkorchid1"),
       lty = 1, lwd = 2)


#Area under the curve for knn------------------------------------------------------------------
AUC_d10<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
            AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
            ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])

AUC_d10









#Hill_vally data
n<-length(test_data$y) #for testset


library(ggplot2)


# Create dataframes for d2, d3, d4, d5, and d10
df_d2 <- data.frame(d = rep("LLO(lambda = 0)", 5),
                    method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                    x_lower = c(MC_d2[1] - (1.96 * sqrt(MC_d2[1] * (1 - MC_d2[1])) / sqrt(n)), 
                                MC_d3[1] - (1.96 * sqrt(MC_d3[1] * (1 - MC_d3[1])) / sqrt(n)), 
                                MC_d4[1] - (1.96 * sqrt(MC_d4[1] * (1 - MC_d4[1])) / sqrt(n)),
                                MC_d5[1] - (1.96 * sqrt(MC_d5[1] * (1 - MC_d5[1])) / sqrt(n)), 
                                MC_d10[1] - (1.96 * sqrt(MC_d10[1] * (1 - MC_d10[1])) / sqrt(n))),
                    x = c(MC_d2[1], MC_d3[1], MC_d4[1], MC_d5[1], MC_d10[1]),
                    x_upper = c(MC_d2[1] + (1.96 * sqrt(MC_d2[1] * (1 - MC_d2[1])) / sqrt(n)), 
                                MC_d3[1] + (1.96 * sqrt(MC_d3[1] * (1 - MC_d3[1])) / sqrt(n)), 
                                MC_d4[1] + (1.96 * sqrt(MC_d4[1] * (1 - MC_d4[1])) / sqrt(n)),
                                MC_d5[1] + (1.96 * sqrt(MC_d5[1] * (1 - MC_d5[1])) / sqrt(n)), 
                                MC_d10[1] + (1.96 * sqrt(MC_d10[1] * (1 - MC_d10[1])) / sqrt(n))))

df_d3 <- data.frame(d = rep("LLO(lambda > 0)", 5),
                    method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                    x_lower = c(MC_d2[2] - (1.96 * sqrt(MC_d2[2] * (1 - MC_d2[2])) / sqrt(n)), 
                                MC_d3[2] - (1.96 * sqrt(MC_d3[2] * (1 - MC_d3[2])) / sqrt(n)), 
                                MC_d4[2] - (1.96 * sqrt(MC_d4[2] * (1 - MC_d4[2])) / sqrt(n)),
                                MC_d5[2] - (1.96 * sqrt(MC_d5[2] * (1 - MC_d5[2])) / sqrt(n)), 
                                MC_d10[2] - (1.96 * sqrt(MC_d10[2] * (1 - MC_d10[2])) / sqrt(n))),
                    x = c(MC_d2[2], MC_d3[2], MC_d4[2], MC_d5[2], MC_d10[2]),
                    x_upper = c(MC_d2[2] + (1.96 * sqrt(MC_d2[2] * (1 - MC_d2[2])) / sqrt(n)), 
                                MC_d3[2] + (1.96 * sqrt(MC_d3[2] * (1 - MC_d3[2])) / sqrt(n)), 
                                MC_d4[2] + (1.96 * sqrt(MC_d4[2] * (1 - MC_d4[2])) / sqrt(n)),
                                MC_d5[2] + (1.96 * sqrt(MC_d5[2] * (1 - MC_d5[2])) / sqrt(n)), 
                                MC_d10[2] + (1.96 * sqrt(MC_d10[2] * (1 - MC_d10[2])) / sqrt(n))))

df_d4 <- data.frame(d = rep("SAVE", 5),
                    method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                    x_lower = c(MC_d2[3] - (1.96 * sqrt(MC_d2[3] * (1 - MC_d2[3])) / sqrt(n)), 
                                MC_d3[3] - (1.96 * sqrt(MC_d3[3] * (1 - MC_d3[3])) / sqrt(n)), 
                                MC_d4[3] - (1.96 * sqrt(MC_d4[3] * (1 - MC_d4[3])) / sqrt(n)),
                                MC_d5[3] - (1.96 * sqrt(MC_d5[3] * (1 - MC_d5[3])) / sqrt(n)), 
                                MC_d10[3] - (1.96 * sqrt(MC_d10[3] * (1 - MC_d10[3])) / sqrt(n))),
                    x = c(MC_d2[3], MC_d3[3], MC_d4[3], MC_d5[3], MC_d10[3]),
                    x_upper = c(MC_d2[3] + (1.96 * sqrt(MC_d2[3] * (1 - MC_d2[3])) / sqrt(n)), 
                                MC_d3[3] + (1.96 * sqrt(MC_d3[3] * (1 - MC_d3[3])) / sqrt(n)), 
                                MC_d4[3] + (1.96 * sqrt(MC_d4[3] * (1 - MC_d4[3])) / sqrt(n)),
                                MC_d5[3] + (1.96 * sqrt(MC_d5[3] * (1 - MC_d5[3])) / sqrt(n)), 
                                MC_d10[3] + (1.96 * sqrt(MC_d10[3] * (1 - MC_d10[3])) / sqrt(n))))

df_d5 <- data.frame(d = rep("PHD", 5),
                    method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                    x_lower = c(MC_d2[4] - (1.96 * sqrt(MC_d2[4] * (1 - MC_d2[4])) / sqrt(n)), 
                                MC_d3[4] - (1.96 * sqrt(MC_d3[4] * (1 - MC_d3[4])) / sqrt(n)), 
                                MC_d4[4] - (1.96 * sqrt(MC_d4[4] * (1 - MC_d4[4])) / sqrt(n)),
                                MC_d5[4] - (1.96 * sqrt(MC_d5[4] * (1 - MC_d5[4])) / sqrt(n)), 
                                MC_d10[4] - (1.96 * sqrt(MC_d10[4] * (1 - MC_d10[4])) / sqrt(n))),
                    x = c(MC_d2[4], MC_d3[4], MC_d4[4], MC_d5[4], MC_d10[4]),
                    x_upper = c(MC_d2[4] + (1.96 * sqrt(MC_d2[4] * (1 - MC_d2[4])) / sqrt(n)), 
                                MC_d3[4] + (1.96 * sqrt(MC_d3[4] * (1 - MC_d3[4])) / sqrt(n)), 
                                MC_d4[4] + (1.96 * sqrt(MC_d4[4] * (1 - MC_d4[4])) / sqrt(n)),
                                MC_d5[4] + (1.96 * sqrt(MC_d5[4] * (1 - MC_d5[4])) / sqrt(n)), 
                                MC_d10[4] + (1.96 * sqrt(MC_d10[4] * (1 - MC_d10[4])) / sqrt(n))))

df_d10 <- data.frame(d = rep("POTD", 5),
                     method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                     x_lower = c(MC_d2[5] - (1.96 * sqrt(MC_d2[5] * (1 - MC_d2[5])) / sqrt(n)), 
                                 MC_d3[5] - (1.96 * sqrt(MC_d3[5] * (1 - MC_d3[5])) / sqrt(n)), 
                                 MC_d4[5] - (1.96 * sqrt(MC_d4[5] * (1 - MC_d4[5])) / sqrt(n)),
                                 MC_d5[5] - (1.96 * sqrt(MC_d5[5] * (1 - MC_d5[5])) / sqrt(n)), 
                                 MC_d10[5] - (1.96 * sqrt(MC_d10[5] * (1 - MC_d10[5])) / sqrt(n))),
                     x = c(MC_d2[5], MC_d3[5], MC_d4[5], MC_d5[5], MC_d10[5]),
                     x_upper = c(MC_d2[5] + (1.96 * sqrt(MC_d2[5] * (1 - MC_d2[5])) / sqrt(n)), 
                                 MC_d3[5] + (1.96 * sqrt(MC_d3[5] * (1 - MC_d3[5])) / sqrt(n)), 
                                 MC_d4[5] + (1.96 * sqrt(MC_d4[5] * (1 - MC_d4[5])) / sqrt(n)),
                                 MC_d5[5] + (1.96 * sqrt(MC_d5[5] * (1 - MC_d5[5])) / sqrt(n)), 
                                 MC_d10[5] + (1.96 * sqrt(MC_d10[5] * (1 - MC_d10[5])) / sqrt(n))))

# Combine dataframes
df_combined <- rbind(df_d2, df_d3, df_d4, df_d5, df_d10)

# Define the order of methods
method_order <- c("d=2", "d=3", "d=4", "d=5", "d=10")

# Convert "method" to ordered factor with custom levels
df_combined$method <- factor(df_combined$method, levels = method_order, ordered = TRUE)

# Reorder the levels of "d" variable
df_combined$d <- factor(df_combined$d, levels = c("LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"))

# Plot
ggplot(df_combined, aes(x = method, y = x, group = d, color = d)) +
  geom_point(position = position_dodge(width = 0.1), size = 2) +
  geom_line(position = position_dodge(width = 0.1), size = 0.5, alpha = 0.7) + # Adjust alpha for line transparency
  geom_errorbar(aes(ymin = x_lower, ymax = x_upper), position = position_dodge(width = 0.1), width = 0.2, alpha = 0.5) +  # Add error bars with adjusted alpha
  labs(x = "Dimensions", y = "Miscl. Risk", title = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "top",  # Display legend on top
        legend.key.size = unit(0.5, "lines")) +  # Set smaller legend key size
  scale_color_discrete(name = "Model", breaks = c("LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"))







###################################################################
##################################################################
### Brest cancer data-------------------
###################################################################
##################################################################







par(mar = c(4, 4, 2, 0.5)) 
par(mfrow=c(2,3))
ntree=500
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
#
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

svd_logistic <- svd(coef.mat_logistic)
svd_lasso <- svd(coef.mat_lasso)
##
#compititors
save.fit =dr(train_data$y~.,data=train_data[,-ncol(train_data)], method="save")
phd.fit = dr(as.numeric(train_data$y)~.,train_data[,-ncol(train_data)], method="phdy")
potd.fit<-potd(X=as.matrix(train_data[,-ncol(train_data)]), y=train_data$y, ndim=ncol(train_data[,-ncol(train_data)]))

###############



d=2


Vk_logistic <- svd_logistic$v[, 1:d]
Vk_lasso <- svd_lasso$v[,  1:d]
Vk_save <- save.fit$evectors[,  1:d]
Vk_phd <- phd.fit$evectors[,  1:d]
Vk_potd <- potd.fit[,  1:d]
#data for prediction-----------------------------------------------------


######Transform for logistic and Lasso 
x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_save
x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_phd
x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
colnames(x_train_transformed_save) <- paste0("PC", 1:d)
colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
colnames(x_train_transformed_potd) <- paste0("PC", 1:d)

x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_save
x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_phd
x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
colnames(x_test_transformed_save) <- paste0("PC", 1:d)
colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
# mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])

###KNN-------------------------------------------------------------------
k_range <- 5 # Example range of k values: 1, 3, 5, 7, 9

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
knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_save<- round(end.time - start.time,2)
#knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)


#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_phd <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)

#
#
start.time <- Sys.time()
#knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
end.time <- Sys.time()
time.taken_potd <- round(end.time - start.time,2)
#knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)



knn_time_d2<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
names(knn_time_d2)<- c("logistic","lasso", "save","phd","potd")
print(knn_time_d2)


# plot(rf_full)
# plot(rf_logistic, add=TRUE)
# plot(rf_lasso, add=TRUE)
# rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
# legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)

#KNNprediction------------------------------------------------------------------------------

predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
#Confusion matrix  (KNN)------------------------------------
conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
#F1 score-----------------------
# F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
# names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(F1_knn,3)
#accuracy_knn-----------------------------
# accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
# names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
# round(accuracy_knn,3)
#AM risk------------------------------------------
AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
AM_d2<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
names(AM_d2)<-  c("logistic","lasso", "save", "phd", "potd")
round(AM_d2,3)
##
#Missclassification  rate------------------------------------------

MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
MC_knn_lasso<-1- conf_knn_lasso$overall[1]
MC_knn_save<-1 - conf_knn_save$overall[1] 
MC_knn_phd<-1 - conf_knn_phd$overall[1] 
MC_knn_potd<-1 - conf_knn_potd$overall[1] 
MC_d2<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
names(MC_d2)<-  c("logistic","lasso", "save", "phd", "potd")
round(MC_d2,3)



F1_d2<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
names(F1_d2)<- c( "logistic","lasso", "save","phd","potd")



######################################
###ROC CURVE------------------------------------------------------------------

#KNN------------------------------------------
pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
# Prediction for the ROC------------------------------
#KNN----------------------------
pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
#Performance----------------------------

#KNN--------------------------
perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")
#Plot------------------------------------

#ROC
plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 2")), lty=1, lwd=2)
rect(par("usr")[1], par("usr")[3],
     par("usr")[2], par("usr")[4],
     col = "#ebebeb")

# Add white grid
grid(nx = NULL, ny = NULL,
     col = "gray", lwd = 1)
plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
# legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
#                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
#                                 "POTD" ),
#        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)

#Area under the curve for knn------------------------------------------------------------------
 AUC_d2<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
            AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
            ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])
 
 AUC_d2


 
 
 ###############
 
 
 
 d=3
 
 
 Vk_logistic <- svd_logistic$v[, 1:d]
 Vk_lasso <- svd_lasso$v[,  1:d]
 Vk_save <- save.fit$evectors[,  1:d]
 Vk_phd <- phd.fit$evectors[,  1:d]
 Vk_potd <- potd.fit[,  1:d]
 #data for prediction-----------------------------------------------------
 
 
 ######Transform for logistic and Lasso 
 x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
 x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
 x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_save
 x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_phd
 x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
 colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_train_transformed_save) <- paste0("PC", 1:d)
 colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_train_transformed_potd) <- paste0("PC", 1:d)
 
 x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
 x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
 x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_save
 x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_phd
 x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
 colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_test_transformed_save) <- paste0("PC", 1:d)
 colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
 # mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])

 
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
 knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_save<- round(end.time - start.time,2)
 #knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_phd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 #
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_potd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 
 knn_time_d3<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
 names(knn_time_d3)<- c("logistic","lasso", "save","phd","potd")
 print(knn_time_d3)

 # plot(rf_full)
 # plot(rf_logistic, add=TRUE)
 # plot(rf_lasso, add=TRUE)
 # rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
 # legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)
 
 #KNNprediction------------------------------------------------------------------------------
 
 predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
 predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
 predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
 predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
 predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
 #Confusion matrix  (KNN)------------------------------------
 conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 #F1 score-----------------------
 # F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
 # names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(F1_knn,3)
 #accuracy_knn-----------------------------
 # accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
 # names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(accuracy_knn,3)
 #AM risk------------------------------------------
 AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
 AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
 AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
 AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
 AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
 AM_d3<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
 names(AM_d3)<-  c("logistic","lasso", "save", "phd", "potd")
 round(AM_d3,3)
 ##
 #Missclassification  rate------------------------------------------
 
 MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
 MC_knn_lasso<-1- conf_knn_lasso$overall[1]
 MC_knn_save<-1 - conf_knn_save$overall[1] 
 MC_knn_phd<-1 - conf_knn_phd$overall[1] 
 MC_knn_potd<-1 - conf_knn_potd$overall[1] 
 MC_d3<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
 names(MC_d3)<-  c("logistic","lasso", "save", "phd", "potd")
 round(MC_d3,3)
 
 
 
 
 F1_d3<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
 names(F1_d3)<- c( "logistic","lasso", "save","phd","potd")
 
 ######################################
 ###ROC CURVE------------------------------------------------------------------
 
 #KNN------------------------------------------
 pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
 pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
 pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
 pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
 pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
 # Prediction for the ROC------------------------------
 #KNN----------------------------
 pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
 pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
 pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
 pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
 pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
 #Performance----------------------------
 
 #KNN--------------------------
 perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
 perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
 perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
 perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
 perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")
 #Plot------------------------------------
 #ROC
 plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 3")), lty=1, lwd=2)
 rect(par("usr")[1], par("usr")[3],
      par("usr")[2], par("usr")[4],
      col = "#ebebeb")
 
 # Add white grid
 grid(nx = NULL, ny = NULL,
      col = "gray", lwd = 1)
 plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
 plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
 plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
 plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
 plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
 abline(a=0,b=1,lwd=2,lty=2,col="gray")
 # legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
 #                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
 #                                 "POTD" ),
 #        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)
 
 #Area under the curve for knn------------------------------------------------------------------
 AUC_d3<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
            AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
            ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])
 
 AUC_d3
 
 
 
 ###############
 
 
 
 d=4
 
 
 Vk_logistic <- svd_logistic$v[, 1:d]
 Vk_lasso <- svd_lasso$v[,  1:d]
 Vk_save <- save.fit$evectors[,  1:d]
 Vk_phd <- phd.fit$evectors[,  1:d]
 Vk_potd <- potd.fit[,  1:d]
 #data for prediction-----------------------------------------------------
 
 
 ######Transform for logistic and Lasso 
 x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
 x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
 x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_save
 x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_phd
 x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
 colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_train_transformed_save) <- paste0("PC", 1:d)
 colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_train_transformed_potd) <- paste0("PC", 1:d)
 
 x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
 x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
 x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_save
 x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_phd
 x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
 colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_test_transformed_save) <- paste0("PC", 1:d)
 colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
 # mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])
 
 ###KNN-------------------------------------------------------------------
 k_range <- 10 # Example range of k values: 1, 3, 5, 7, 9
 
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
 knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_save<- round(end.time - start.time,2)
 #knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_phd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 #
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_potd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 
 knn_time_d4<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
 names(knn_time_d4)<- c("logistic","lasso", "save","phd","potd")
 print(knn_time_d4)
 
 # plot(rf_full)
 # plot(rf_logistic, add=TRUE)
 # plot(rf_lasso, add=TRUE)
 # rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
 # legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)
 
 #KNNprediction------------------------------------------------------------------------------
 
 predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
 predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
 predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
 predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
 predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
 #Confusion matrix  (KNN)------------------------------------
 conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 #F1 score-----------------------
 # F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
 # names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(F1_knn,3)
 #accuracy_knn-----------------------------
 # accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
 # names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(accuracy_knn,3)
 #AM risk------------------------------------------
 AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
 AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
 AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
 AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
 AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
 AM_d4<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
 names(AM_d4)<-  c("logistic","lasso", "save", "phd", "potd")
 round(AM_d4,3)
 ##
 #Missclassification  rate------------------------------------------
 
 MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
 MC_knn_lasso<-1- conf_knn_lasso$overall[1]
 MC_knn_save<-1 - conf_knn_save$overall[1] 
 MC_knn_phd<-1 - conf_knn_phd$overall[1] 
 MC_knn_potd<-1 - conf_knn_potd$overall[1] 
 MC_d4<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
 names(MC_d4)<-  c("logistic","lasso", "save", "phd", "potd")
 round(MC_d4,3)
 
 
 
 
 F1_d4<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
 names(F1_d4)<- c( "logistic","lasso", "save","phd","potd")
 
 ######################################
 ###ROC CURVE------------------------------------------------------------------
 
 #KNN------------------------------------------
 pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
 pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
 pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
 pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
 pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
 # Prediction for the ROC------------------------------
 #KNN----------------------------
 pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
 pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
 pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
 pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
 pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
 #Performance----------------------------
 
 #KNN--------------------------
 perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
 perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
 perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
 perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
 perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")
 #Plot------------------------------------
 #ROC
 plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 4")), lty=1, lwd=2)
 rect(par("usr")[1], par("usr")[3],
      par("usr")[2], par("usr")[4],
      col = "#ebebeb")
 
 # Add white grid
 grid(nx = NULL, ny = NULL,
      col = "gray", lwd = 1)
 plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
 plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
 plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
 plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
 plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
 abline(a=0,b=1,lwd=2,lty=2,col="gray")
 # legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
 #                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
 #                                 "POTD" ),
 #        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)
 
 #Area under the curve for knn------------------------------------------------------------------
 AUC_d4<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
            AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
            ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])
 
 AUC_d4
 
 
 
 ###############
 
 
 
 d=5
 
 
 Vk_logistic <- svd_logistic$v[, 1:d]
 Vk_lasso <- svd_lasso$v[,  1:d]
 Vk_save <- save.fit$evectors[,  1:d]
 Vk_phd <- phd.fit$evectors[,  1:d]
 Vk_potd <- potd.fit[,  1:d]
 #data for prediction-----------------------------------------------------
 
 
 ######Transform for logistic and Lasso 
 x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
 x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
 x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_save
 x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_phd
 x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
 colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_train_transformed_save) <- paste0("PC", 1:d)
 colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_train_transformed_potd) <- paste0("PC", 1:d)
 
 x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
 x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
 x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_save
 x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_phd
 x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
 colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_test_transformed_save) <- paste0("PC", 1:d)
 colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
 # mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])
 
 ###KNN-------------------------------------------------------------------
 
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
 knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_save<- round(end.time - start.time,2)
 #knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_phd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 #
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_potd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 
 knn_time_d5<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
 names(knn_time_d5)<- c("logistic","lasso", "save","phd","potd")
 print(knn_time_d5)
 
 # plot(rf_full)
 # plot(rf_logistic, add=TRUE)
 # plot(rf_lasso, add=TRUE)
 # rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
 # legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)
 
 #KNNprediction------------------------------------------------------------------------------
 
 predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
 predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
 predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
 predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
 predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
 #Confusion matrix  (KNN)------------------------------------
 conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 #F1 score-----------------------
 # F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
 # names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(F1_knn,3)
 #accuracy_knn-----------------------------
 # accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
 # names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(accuracy_knn,3)
 #AM risk------------------------------------------
 AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
 AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
 AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
 AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
 AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
 AM_d5<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
 names(AM_d5)<-  c("logistic","lasso", "save", "phd", "potd")
 round(AM_d5,3)
 ##
 #Missclassification  rate------------------------------------------
 
 MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
 MC_knn_lasso<-1- conf_knn_lasso$overall[1]
 MC_knn_save<-1 - conf_knn_save$overall[1] 
 MC_knn_phd<-1 - conf_knn_phd$overall[1] 
 MC_knn_potd<-1 - conf_knn_potd$overall[1] 
 MC_d5<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
 names(MC_d5)<-  c("logistic","lasso", "save", "phd", "potd")
 round(MC_d5,3)
 
 
 
 F1_d5<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
 names(F1_d5)<- c( "logistic","lasso", "save","phd","potd")
 
 ######################################
 ###ROC CURVE------------------------------------------------------------------
 
 #KNN------------------------------------------
 pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
 pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
 pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
 pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
 pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
 # Prediction for the ROC------------------------------
 #KNN----------------------------
 pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
 pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
 pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
 pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
 pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
 #Performance----------------------------
 
 #KNN--------------------------
 perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
 perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
 perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
 perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
 perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")


 #ROC
 plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 5")), lty=1, lwd=2)
 rect(par("usr")[1], par("usr")[3],
      par("usr")[2], par("usr")[4],
      col = "#ebebeb")
 
 # Add white grid
 grid(nx = NULL, ny = NULL,
      col = "gray", lwd = 1)
 plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
 plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
 plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
 plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
 plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
 abline(a=0,b=1,lwd=2,lty=2,col="gray")
 # legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
 #                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
 #                                 "POTD" ),
 #        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)
 
 #Area under the curve for knn------------------------------------------------------------------
 AUC_d5<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
            AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
            ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])
 
 AUC_d5
 
 
 
 ###############
 
 
 
 d=10
 
 
 Vk_logistic <- svd_logistic$v[, 1:d]
 Vk_lasso <- svd_lasso$v[,  1:d]
 Vk_save <- save.fit$evectors[,  1:d]
 Vk_phd <- phd.fit$evectors[,  1:d]
 Vk_potd <- potd.fit[,  1:d]
 #data for prediction-----------------------------------------------------
 
 
 ######Transform for logistic and Lasso 
 x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
 x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
 x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_save
 x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_phd
 x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
 colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_train_transformed_save) <- paste0("PC", 1:d)
 colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_train_transformed_potd) <- paste0("PC", 1:d)
 
 x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
 x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
 x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_save
 x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_phd
 x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
 colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_test_transformed_save) <- paste0("PC", 1:d)
 colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
 # mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])
 
 ###KNN-------------------------------------------------------------------
 
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
 knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_save<- round(end.time - start.time,2)
 #knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_phd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 #
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_potd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 
 knn_time_d10<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
 names(knn_time_d10)<- c("logistic","lasso", "save","phd","potd")
 print(knn_time_d10)
 
 # plot(rf_full)
 # plot(rf_logistic, add=TRUE)
 # plot(rf_lasso, add=TRUE)
 # rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
 # legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)
 
 #KNNprediction------------------------------------------------------------------------------
 
 predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
 predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
 predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
 predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
 predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
 #Confusion matrix  (KNN)------------------------------------
 conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 #F1 score-----------------------
 # F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
 # names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(F1_knn,3)
 #accuracy_knn-----------------------------
 # accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
 # names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(accuracy_knn,3)
 #AM risk------------------------------------------
 AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
 AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
 AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
 AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
 AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
 AM_d10<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
 names(AM_d10)<-  c("logistic","lasso", "save", "phd", "potd")
 round(AM_d10,3)
 ##
 #Missclassification  rate------------------------------------------
 
 MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
 MC_knn_lasso<-1- conf_knn_lasso$overall[1]
 MC_knn_save<-1 - conf_knn_save$overall[1] 
 MC_knn_phd<-1 - conf_knn_phd$overall[1] 
 MC_knn_potd<-1 - conf_knn_potd$overall[1] 
 MC_d10<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
 names(MC_d10)<-  c("logistic","lasso", "save", "phd", "potd")
 round(MC_d10,3)
 
 
 F1_d10<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
 names(F1_d10)<- c( "logistic","lasso", "save","phd","potd")
 
 
 ######################################
 ###ROC CURVE------------------------------------------------------------------
 
 #KNN------------------------------------------
 pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
 pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
 pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
 pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
 pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
 # Prediction for the ROC------------------------------
 #KNN----------------------------
 pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
 pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
 pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
 pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
 pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
 #Performance----------------------------
 
 #KNN--------------------------
 perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
 perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
 perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
 perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
 perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")
 #Plot------------------------------------

 #Knn----
 #ROC
 plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 10")), lty=1, lwd=2)
 rect(par("usr")[1], par("usr")[3],
      par("usr")[2], par("usr")[4],
      col = "#ebebeb")
 
 # Add white grid
 grid(nx = NULL, ny = NULL,
      col = "gray", lwd = 1)
 plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
 plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
 plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
 plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
 plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
 abline(a=0,b=1,lwd=2,lty=2,col="gray")
 
 # legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
 #                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
 #                                 "POTD" ),
 #        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)
 
 
 plot(0, type = "n", xlim = c(0, 1), ylim = c(0, 1), xlab = "", ylab = "", main=expression(paste(  "Legends")))
 rect(par("usr")[1], par("usr")[3],
      par("usr")[2], par("usr")[4],
      col = "#ebebeb")
 
 # Add white grid
 grid(nx = NULL, ny = NULL,
      col = "gray", lwd = 1)
 # Add legends
 legend("center",
        legend = c(expression(paste("LLO(", lambda, "= 0)")),
                   expression(paste("LLO(", lambda, "> 0)")),
                   "SAVE", "PHD", "POTD"),
        col = c("darkorange", "olivedrab", "springgreen", "cyan2", "darkorchid1"),
        lty = 1, lwd = 2)
 
 
 
 #Area under the curve for knn------------------------------------------------------------------
 AUC_d10<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
            AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
            ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])
 
 AUC_d10

 
 
 

 
 
 #Hill_vally data
 n<-length(test_data$y) #for testset
 

 
 # Create dataframes for d2, d3, d4, d5, and d10
 df_d2 <- data.frame(d = rep("LLO(lambda = 0)", 5),
                     method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                     x_lower = c(MC_d2[1] - (1.96 * sqrt(MC_d2[1] * (1 - MC_d2[1])) / sqrt(n)), 
                                 MC_d3[1] - (1.96 * sqrt(MC_d3[1] * (1 - MC_d3[1])) / sqrt(n)), 
                                 MC_d4[1] - (1.96 * sqrt(MC_d4[1] * (1 - MC_d4[1])) / sqrt(n)),
                                 MC_d5[1] - (1.96 * sqrt(MC_d5[1] * (1 - MC_d5[1])) / sqrt(n)), 
                                 MC_d10[1] - (1.96 * sqrt(MC_d10[1] * (1 - MC_d10[1])) / sqrt(n))),
                     x = c(MC_d2[1], MC_d3[1], MC_d4[1], MC_d5[1], MC_d10[1]),
                     x_upper = c(MC_d2[1] + (1.96 * sqrt(MC_d2[1] * (1 - MC_d2[1])) / sqrt(n)), 
                                 MC_d3[1] + (1.96 * sqrt(MC_d3[1] * (1 - MC_d3[1])) / sqrt(n)), 
                                 MC_d4[1] + (1.96 * sqrt(MC_d4[1] * (1 - MC_d4[1])) / sqrt(n)),
                                 MC_d5[1] + (1.96 * sqrt(MC_d5[1] * (1 - MC_d5[1])) / sqrt(n)), 
                                 MC_d10[1] + (1.96 * sqrt(MC_d10[1] * (1 - MC_d10[1])) / sqrt(n))))
 
 df_d3 <- data.frame(d = rep("LLO(lambda > 0)", 5),
                     method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                     x_lower = c(MC_d2[2] - (1.96 * sqrt(MC_d2[2] * (1 - MC_d2[2])) / sqrt(n)), 
                                 MC_d3[2] - (1.96 * sqrt(MC_d3[2] * (1 - MC_d3[2])) / sqrt(n)), 
                                 MC_d4[2] - (1.96 * sqrt(MC_d4[2] * (1 - MC_d4[2])) / sqrt(n)),
                                 MC_d5[2] - (1.96 * sqrt(MC_d5[2] * (1 - MC_d5[2])) / sqrt(n)), 
                                 MC_d10[2] - (1.96 * sqrt(MC_d10[2] * (1 - MC_d10[2])) / sqrt(n))),
                     x = c(MC_d2[2], MC_d3[2], MC_d4[2], MC_d5[2], MC_d10[2]),
                     x_upper = c(MC_d2[2] + (1.96 * sqrt(MC_d2[2] * (1 - MC_d2[2])) / sqrt(n)), 
                                 MC_d3[2] + (1.96 * sqrt(MC_d3[2] * (1 - MC_d3[2])) / sqrt(n)), 
                                 MC_d4[2] + (1.96 * sqrt(MC_d4[2] * (1 - MC_d4[2])) / sqrt(n)),
                                 MC_d5[2] + (1.96 * sqrt(MC_d5[2] * (1 - MC_d5[2])) / sqrt(n)), 
                                 MC_d10[2] + (1.96 * sqrt(MC_d10[2] * (1 - MC_d10[2])) / sqrt(n))))
 
 df_d4 <- data.frame(d = rep("SAVE", 5),
                     method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                     x_lower = c(MC_d2[3] - (1.96 * sqrt(MC_d2[3] * (1 - MC_d2[3])) / sqrt(n)), 
                                 MC_d3[3] - (1.96 * sqrt(MC_d3[3] * (1 - MC_d3[3])) / sqrt(n)), 
                                 MC_d4[3] - (1.96 * sqrt(MC_d4[3] * (1 - MC_d4[3])) / sqrt(n)),
                                 MC_d5[3] - (1.96 * sqrt(MC_d5[3] * (1 - MC_d5[3])) / sqrt(n)), 
                                 MC_d10[3] - (1.96 * sqrt(MC_d10[3] * (1 - MC_d10[3])) / sqrt(n))),
                     x = c(MC_d2[3], MC_d3[3], MC_d4[3], MC_d5[3], MC_d10[3]),
                     x_upper = c(MC_d2[3] + (1.96 * sqrt(MC_d2[3] * (1 - MC_d2[3])) / sqrt(n)), 
                                 MC_d3[3] + (1.96 * sqrt(MC_d3[3] * (1 - MC_d3[3])) / sqrt(n)), 
                                 MC_d4[3] + (1.96 * sqrt(MC_d4[3] * (1 - MC_d4[3])) / sqrt(n)),
                                 MC_d5[3] + (1.96 * sqrt(MC_d5[3] * (1 - MC_d5[3])) / sqrt(n)), 
                                 MC_d10[3] + (1.96 * sqrt(MC_d10[3] * (1 - MC_d10[3])) / sqrt(n))))
 
 df_d5 <- data.frame(d = rep("PHD", 5),
                     method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                     x_lower = c(MC_d2[4] - (1.96 * sqrt(MC_d2[4] * (1 - MC_d2[4])) / sqrt(n)), 
                                 MC_d3[4] - (1.96 * sqrt(MC_d3[4] * (1 - MC_d3[4])) / sqrt(n)), 
                                 MC_d4[4] - (1.96 * sqrt(MC_d4[4] * (1 - MC_d4[4])) / sqrt(n)),
                                 MC_d5[4] - (1.96 * sqrt(MC_d5[4] * (1 - MC_d5[4])) / sqrt(n)), 
                                 MC_d10[4] - (1.96 * sqrt(MC_d10[4] * (1 - MC_d10[4])) / sqrt(n))),
                     x = c(MC_d2[4], MC_d3[4], MC_d4[4], MC_d5[4], MC_d10[4]),
                     x_upper = c(MC_d2[4] + (1.96 * sqrt(MC_d2[4] * (1 - MC_d2[4])) / sqrt(n)), 
                                 MC_d3[4] + (1.96 * sqrt(MC_d3[4] * (1 - MC_d3[4])) / sqrt(n)), 
                                 MC_d4[4] + (1.96 * sqrt(MC_d4[4] * (1 - MC_d4[4])) / sqrt(n)),
                                 MC_d5[4] + (1.96 * sqrt(MC_d5[4] * (1 - MC_d5[4])) / sqrt(n)), 
                                 MC_d10[4] + (1.96 * sqrt(MC_d10[4] * (1 - MC_d10[4])) / sqrt(n))))
 
 df_d10 <- data.frame(d = rep("POTD", 5),
                      method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                      x_lower = c(MC_d2[5] - (1.96 * sqrt(MC_d2[5] * (1 - MC_d2[5])) / sqrt(n)), 
                                  MC_d3[5] - (1.96 * sqrt(MC_d3[5] * (1 - MC_d3[5])) / sqrt(n)), 
                                  MC_d4[5] - (1.96 * sqrt(MC_d4[5] * (1 - MC_d4[5])) / sqrt(n)),
                                  MC_d5[5] - (1.96 * sqrt(MC_d5[5] * (1 - MC_d5[5])) / sqrt(n)), 
                                  MC_d10[5] - (1.96 * sqrt(MC_d10[5] * (1 - MC_d10[5])) / sqrt(n))),
                      x = c(MC_d2[5], MC_d3[5], MC_d4[5], MC_d5[5], MC_d10[5]),
                      x_upper = c(MC_d2[5] + (1.96 * sqrt(MC_d2[5] * (1 - MC_d2[5])) / sqrt(n)), 
                                  MC_d3[5] + (1.96 * sqrt(MC_d3[5] * (1 - MC_d3[5])) / sqrt(n)), 
                                  MC_d4[5] + (1.96 * sqrt(MC_d4[5] * (1 - MC_d4[5])) / sqrt(n)),
                                  MC_d5[5] + (1.96 * sqrt(MC_d5[5] * (1 - MC_d5[5])) / sqrt(n)), 
                                  MC_d10[5] + (1.96 * sqrt(MC_d10[5] * (1 - MC_d10[5])) / sqrt(n))))
 
 # Combine dataframes
 df_combined <- rbind(df_d2, df_d3, df_d4, df_d5, df_d10)
 
 # Define the order of methods
 method_order <- c("d=2", "d=3", "d=4", "d=5", "d=10")
 
 # Convert "method" to ordered factor with custom levels
 df_combined$method <- factor(df_combined$method, levels = method_order, ordered = TRUE)
 
 # Reorder the levels of "d" variable
 df_combined$d <- factor(df_combined$d, levels = c("LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"))
 
 # Plot
 ggplot(df_combined, aes(x = method, y = x, group = d, color = d)) +
   geom_point(position = position_dodge(width = 0.1), size = 2) +
   geom_line(position = position_dodge(width = 0.1), size = 0.5, alpha = 0.7) + # Adjust alpha for line transparency
   geom_errorbar(aes(ymin = x_lower, ymax = x_upper), position = position_dodge(width = 0.1), width = 0.2, alpha = 0.5) +  # Add error bars with adjusted alpha
   labs(x = "Dimensions", y = "Miscl. Risk", title = "") +
   theme(axis.text.x = element_text(angle = 45, hjust = 1),
         legend.position = "top",  # Display legend on top
         legend.key.size = unit(0.5, "lines")) +  # Set smaller legend key size
   scale_color_discrete(name = "Model", breaks = c("LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"))
 
 
 
 
 
 
 
 
 
 
 


 ###################################################################
 ##################################################################
 ### mice+protein+expression-------------------
 ###################################################################
 ##################################################################
 
 par(mar = c(4, 4, 2, 0.5)) 
 ##
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
 
 svd_logistic <- svd(coef.mat_logistic)
 svd_lasso <- svd(coef.mat_lasso)
 ##
 #compititors
 save.fit =dr(train_data$y~.,data=train_data[,-ncol(train_data)], method="save")
 phd.fit = dr(as.numeric(train_data$y)~.,train_data[,-ncol(train_data)], method="phdy")
 potd.fit<-potd(X=as.matrix(train_data[,-ncol(train_data)]), y=train_data$y, ndim=ncol(train_data[,-ncol(train_data)]))
 
 ###############
 
 
 
 d=2
 
 
 Vk_logistic <- svd_logistic$v[, 1:d]
 Vk_lasso <- svd_lasso$v[,  1:d]
 Vk_save <- save.fit$evectors[,  1:d]
 Vk_phd <- phd.fit$evectors[,  1:d]
 Vk_potd <- potd.fit[,  1:d]
 #data for prediction-----------------------------------------------------
 
 
 ######Transform for logistic and Lasso 
 x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
 x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
 x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)][-1]) %*% Vk_save
 x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)][-1]) %*% Vk_phd
 x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
 colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_train_transformed_save) <- paste0("PC", 1:d)
 colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_train_transformed_potd) <- paste0("PC", 1:d)
 
 x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
 x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
 x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)][-1])  %*% Vk_save
 x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)][-1])  %*% Vk_phd
 x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
 colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_test_transformed_save) <- paste0("PC", 1:d)
 colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
 # mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])
 
 ###KNN-------------------------------------------------------------------
 k_range <- 5 # Example range of k values: 1, 3, 5, 7, 9
 
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
 knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_save<- round(end.time - start.time,2)
 #knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_phd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 #
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_potd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 
 knn_time_d2<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
 names(knn_time_d2)<- c("logistic","lasso", "save","phd","potd")
 print(knn_time_d2)
 
 
 # plot(rf_full)
 # plot(rf_logistic, add=TRUE)
 # plot(rf_lasso, add=TRUE)
 # rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
 # legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)
 
 #KNNprediction------------------------------------------------------------------------------
 
 predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
 predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
 predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
 predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
 predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
 #Confusion matrix  (KNN)------------------------------------
 conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 #F1 score-----------------------
 # F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
 # names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(F1_knn,3)
 #accuracy_knn-----------------------------
 # accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
 # names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(accuracy_knn,3)
 #AM risk------------------------------------------
 AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
 AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
 AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
 AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
 AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
 AM_d2<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
 names(AM_d2)<-  c("logistic","lasso", "save", "phd", "potd")
 round(AM_d2,3)
 ##
 #Missclassification  rate------------------------------------------
 
 MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
 MC_knn_lasso<-1- conf_knn_lasso$overall[1]
 MC_knn_save<-1 - conf_knn_save$overall[1] 
 MC_knn_phd<-1 - conf_knn_phd$overall[1] 
 MC_knn_potd<-1 - conf_knn_potd$overall[1] 
 MC_d2<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
 names(MC_d2)<-  c("logistic","lasso", "save", "phd", "potd")
 round(MC_d2,3)
 
 
 
 F1_d2<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
 names(F1_d2)<- c( "logistic","lasso", "save","phd","potd")
 
 
 
 ######################################
 ###ROC CURVE------------------------------------------------------------------
 
 #KNN------------------------------------------
 pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
 pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
 pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
 pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
 pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
 # Prediction for the ROC------------------------------
 #KNN----------------------------
 pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
 pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
 pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
 pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
 pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
 #Performance----------------------------
 
 #KNN--------------------------
 perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
 perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
 perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
 perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
 perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")
 #Plot------------------------------------
 
 #ROC
 plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 2")), lty=1, lwd=2)
 rect(par("usr")[1], par("usr")[3],
      par("usr")[2], par("usr")[4],
      col = "#ebebeb")
 
 # Add white grid
 grid(nx = NULL, ny = NULL,
      col = "gray", lwd = 1)
 plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
 plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
 plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
 plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
 plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
 abline(a=0,b=1,lwd=2,lty=2,col="gray")
 # legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
 #                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
 #                                 "POTD" ),
 #        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)
 
 #Area under the curve for knn------------------------------------------------------------------
 AUC_d2<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
            AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
            ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])
 
 AUC_d2
 
 
 
 
 ###############
 
 
 
 d=3
 
 
 Vk_logistic <- svd_logistic$v[, 1:d]
 Vk_lasso <- svd_lasso$v[,  1:d]
 Vk_save <- save.fit$evectors[,  1:d]
 Vk_phd <- phd.fit$evectors[,  1:d]
 Vk_potd <- potd.fit[,  1:d]
 #data for prediction-----------------------------------------------------
 
 
 ######Transform for logistic and Lasso 
 x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
 x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
 x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)][-1]) %*% Vk_save
 x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)][-1]) %*% Vk_phd
 x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
 colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_train_transformed_save) <- paste0("PC", 1:d)
 colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_train_transformed_potd) <- paste0("PC", 1:d)
 
 x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
 x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
 x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)][-1])  %*% Vk_save
 x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)][-1])  %*% Vk_phd
 x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
 colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_test_transformed_save) <- paste0("PC", 1:d)
 colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
 # mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])
 
 
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
 knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_save<- round(end.time - start.time,2)
 #knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_phd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 #
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_potd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 
 knn_time_d3<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
 names(knn_time_d3)<- c("logistic","lasso", "save","phd","potd")
 print(knn_time_d3)
 
 # plot(rf_full)
 # plot(rf_logistic, add=TRUE)
 # plot(rf_lasso, add=TRUE)
 # rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
 # legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)
 
 #KNNprediction------------------------------------------------------------------------------
 
 predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
 predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
 predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
 predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
 predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
 #Confusion matrix  (KNN)------------------------------------
 conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 #F1 score-----------------------
 # F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
 # names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(F1_knn,3)
 #accuracy_knn-----------------------------
 # accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
 # names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(accuracy_knn,3)
 #AM risk------------------------------------------
 AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
 AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
 AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
 AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
 AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
 AM_d3<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
 names(AM_d3)<-  c("logistic","lasso", "save", "phd", "potd")
 round(AM_d3,3)
 ##
 #Missclassification  rate------------------------------------------
 
 MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
 MC_knn_lasso<-1- conf_knn_lasso$overall[1]
 MC_knn_save<-1 - conf_knn_save$overall[1] 
 MC_knn_phd<-1 - conf_knn_phd$overall[1] 
 MC_knn_potd<-1 - conf_knn_potd$overall[1] 
 MC_d3<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
 names(MC_d3)<-  c("logistic","lasso", "save", "phd", "potd")
 round(MC_d3,3)
 
 
 
 
 F1_d3<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
 names(F1_d3)<- c( "logistic","lasso", "save","phd","potd")
 
 ######################################
 ###ROC CURVE------------------------------------------------------------------
 
 #KNN------------------------------------------
 pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
 pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
 pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
 pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
 pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
 # Prediction for the ROC------------------------------
 #KNN----------------------------
 pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
 pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
 pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
 pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
 pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
 #Performance----------------------------
 
 #KNN--------------------------
 perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
 perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
 perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
 perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
 perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")
 #Plot------------------------------------
 
 #ROC
 plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 3")), lty=1, lwd=2)
 rect(par("usr")[1], par("usr")[3],
      par("usr")[2], par("usr")[4],
      col = "#ebebeb")
 
 # Add white grid
 grid(nx = NULL, ny = NULL,
      col = "gray", lwd = 1)
 plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
 plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
 plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
 plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
 plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
 abline(a=0,b=1,lwd=2,lty=2,col="gray")
 # legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
 #                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
 #                                 "POTD" ),
 #        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)
 
 #Area under the curve for knn------------------------------------------------------------------
 AUC_d3<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
            AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
            ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])
 
 AUC_d3
 
 
 
 ###############
 
 
 
 d=4
 
 
 Vk_logistic <- svd_logistic$v[, 1:d]
 Vk_lasso <- svd_lasso$v[,  1:d]
 Vk_save <- save.fit$evectors[,  1:d]
 Vk_phd <- phd.fit$evectors[,  1:d]
 Vk_potd <- potd.fit[,  1:d]
 #data for prediction-----------------------------------------------------
 
 
 ######Transform for logistic and Lasso 
 x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
 x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
 x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)][-1]) %*% Vk_save
 x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)][-1]) %*% Vk_phd
 x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
 colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_train_transformed_save) <- paste0("PC", 1:d)
 colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_train_transformed_potd) <- paste0("PC", 1:d)
 
 x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
 x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
 x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)][-1])  %*% Vk_save
 x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)][-1])  %*% Vk_phd
 x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
 colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_test_transformed_save) <- paste0("PC", 1:d)
 colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
 # mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])
 
 ###KNN-------------------------------------------------------------------
 
 
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
 knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_save<- round(end.time - start.time,2)
 #knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_phd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 #
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_potd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 
 knn_time_d4<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
 names(knn_time_d4)<- c("logistic","lasso", "save","phd","potd")
 print(knn_time_d4)
 
 # plot(rf_full)
 # plot(rf_logistic, add=TRUE)
 # plot(rf_lasso, add=TRUE)
 # rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
 # legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)
 
 #KNNprediction------------------------------------------------------------------------------
 
 predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
 predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
 predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
 predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
 predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
 #Confusion matrix  (KNN)------------------------------------
 conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 #F1 score-----------------------
 # F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
 # names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(F1_knn,3)
 #accuracy_knn-----------------------------
 # accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
 # names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(accuracy_knn,3)
 #AM risk------------------------------------------
 AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
 AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
 AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
 AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
 AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
 AM_d4<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
 names(AM_d4)<-  c("logistic","lasso", "save", "phd", "potd")
 round(AM_d4,3)
 ##
 #Missclassification  rate------------------------------------------
 
 MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
 MC_knn_lasso<-1- conf_knn_lasso$overall[1]
 MC_knn_save<-1 - conf_knn_save$overall[1] 
 MC_knn_phd<-1 - conf_knn_phd$overall[1] 
 MC_knn_potd<-1 - conf_knn_potd$overall[1] 
 MC_d4<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
 names(MC_d4)<-  c("logistic","lasso", "save", "phd", "potd")
 round(MC_d4,3)
 
 
 
 
 F1_d4<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
 names(F1_d4)<- c( "logistic","lasso", "save","phd","potd")
 
 ######################################
 ###ROC CURVE------------------------------------------------------------------
 
 #KNN------------------------------------------
 pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
 pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
 pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
 pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
 pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
 # Prediction for the ROC------------------------------
 #KNN----------------------------
 pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
 pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
 pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
 pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
 pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
 #Performance----------------------------
 
 #KNN--------------------------
 perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
 perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
 perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
 perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
 perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")
 #Plot------------------------------------
 #ROC
 plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 4")), lty=1, lwd=2)
 rect(par("usr")[1], par("usr")[3],
      par("usr")[2], par("usr")[4],
      col = "#ebebeb")
 
 # Add white grid
 grid(nx = NULL, ny = NULL,
      col = "gray", lwd = 1)
 plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
 plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
 plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
 plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
 plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
 abline(a=0,b=1,lwd=2,lty=2,col="gray")
 # legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
 #                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
 #                                 "POTD" ),
 #        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)
 #Area under the curve for knn------------------------------------------------------------------
 AUC_d4<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
            AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
            ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])
 
 AUC_d4
 
 
 
 ###############
 
 
 
 d=5
 
 
 Vk_logistic <- svd_logistic$v[, 1:d]
 Vk_lasso <- svd_lasso$v[,  1:d]
 Vk_save <- save.fit$evectors[,  1:d]
 Vk_phd <- phd.fit$evectors[,  1:d]
 Vk_potd <- potd.fit[,  1:d]
 #data for prediction-----------------------------------------------------
 
 
 ######Transform for logistic and Lasso 
 x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
 x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
 x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)][-1]) %*% Vk_save
 x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)][-1]) %*% Vk_phd
 x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
 colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_train_transformed_save) <- paste0("PC", 1:d)
 colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_train_transformed_potd) <- paste0("PC", 1:d)
 
 x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
 x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
 x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)][-1])  %*% Vk_save
 x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)][-1])  %*% Vk_phd
 x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
 colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_test_transformed_save) <- paste0("PC", 1:d)
 colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
 # mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])
 
 ###KNN-------------------------------------------------------------------
 
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
 knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_save<- round(end.time - start.time,2)
 #knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_phd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 #
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_potd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 
 knn_time_d5<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
 names(knn_time_d5)<- c("logistic","lasso", "save","phd","potd")
 print(knn_time_d5)
 
 # plot(rf_full)
 # plot(rf_logistic, add=TRUE)
 # plot(rf_lasso, add=TRUE)
 # rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
 # legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)
 
 #KNNprediction------------------------------------------------------------------------------
 
 predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
 predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
 predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
 predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
 predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
 #Confusion matrix  (KNN)------------------------------------
 conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 #F1 score-----------------------
 # F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
 # names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(F1_knn,3)
 #accuracy_knn-----------------------------
 # accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
 # names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(accuracy_knn,3)
 #AM risk------------------------------------------
 AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
 AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
 AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
 AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
 AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
 AM_d5<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
 names(AM_d5)<-  c("logistic","lasso", "save", "phd", "potd")
 round(AM_d5,3)
 ##
 #Missclassification  rate------------------------------------------
 
 MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
 MC_knn_lasso<-1- conf_knn_lasso$overall[1]
 MC_knn_save<-1 - conf_knn_save$overall[1] 
 MC_knn_phd<-1 - conf_knn_phd$overall[1] 
 MC_knn_potd<-1 - conf_knn_potd$overall[1] 
 MC_d5<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
 names(MC_d5)<-  c("logistic","lasso", "save", "phd", "potd")
 round(MC_d5,3)
 
 
 
 F1_d5<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
 names(F1_d5)<- c( "logistic","lasso", "save","phd","potd")
 
 ######################################
 ###ROC CURVE------------------------------------------------------------------
 
 #KNN------------------------------------------
 pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
 pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
 pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
 pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
 pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
 # Prediction for the ROC------------------------------
 #KNN----------------------------
 pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
 pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
 pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
 pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
 pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
 #Performance----------------------------
 
 #KNN--------------------------
 perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
 perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
 perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
 perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
 perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")
 #Plot------------------------------------
 #ROC
 plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 5")), lty=1, lwd=2)
 rect(par("usr")[1], par("usr")[3],
      par("usr")[2], par("usr")[4],
      col = "#ebebeb")
 
 # Add white grid
 grid(nx = NULL, ny = NULL,
      col = "gray", lwd = 1)
 plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
 plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
 plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
 plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
 plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
 abline(a=0,b=1,lwd=2,lty=2,col="gray")
 # legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
 #                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
 #                                 "POTD" ),
 #        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)
 
 #Area under the curve for knn------------------------------------------------------------------
 AUC_d5<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
            AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
            ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])
 
 AUC_d5
 
 
 
 ###############
 
 
 
 d=10
 
 
 Vk_logistic <- svd_logistic$v[, 1:d]
 Vk_lasso <- svd_lasso$v[,  1:d]
 Vk_save <- save.fit$evectors[,  1:d]
 Vk_phd <- phd.fit$evectors[,  1:d]
 Vk_potd <- potd.fit[,  1:d]
 #data for prediction-----------------------------------------------------
 
 
 ######Transform for logistic and Lasso 
 x_train_transformed_logistic <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_logistic
 x_train_transformed_lasso <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_lasso
 x_train_transformed_save <- as.matrix(train_data[, -ncol(train_data)][-1]) %*% Vk_save
 x_train_transformed_phd <- as.matrix(train_data[, -ncol(train_data)][-1]) %*% Vk_phd
 x_train_transformed_potd <- as.matrix(train_data[, -ncol(train_data)]) %*% Vk_potd
 colnames(x_train_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_train_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_train_transformed_save) <- paste0("PC", 1:d)
 colnames(x_train_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_train_transformed_potd) <- paste0("PC", 1:d)
 
 x_test_transformed_logistic <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_logistic
 x_test_transformed_lasso <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_lasso
 x_test_transformed_save <- as.matrix(test_data[, -ncol(test_data)][-1])  %*% Vk_save
 x_test_transformed_phd <- as.matrix(test_data[, -ncol(test_data)][-1])  %*% Vk_phd
 x_test_transformed_potd <- as.matrix(test_data[, -ncol(test_data)])  %*% Vk_potd
 colnames(x_test_transformed_logistic) <- paste0("PC", 1:d)
 colnames(x_test_transformed_lasso) <- paste0("PC", 1:d)
 colnames(x_test_transformed_save) <- paste0("PC", 1:d)
 colnames(x_test_transformed_phd) <- paste0("PC", 1:d)
 colnames(x_test_transformed_potd) <- paste0("PC", 1:d)
 # mydata_logistic<- cbind( x_train_transformed_logistic,train_data[ncol(train_data)])
 
 ###KNN-------------------------------------------------------------------
 
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
 knn_save <- train(x =x_train_transformed_save, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_save<- round(end.time - start.time,2)
 #knn_logistic_full_prob <- class::knn(train = x_train_transformed_logistic_full, test = x_test_transformed_logistic_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_phd <- train(x =x_train_transformed_phd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_phd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 #
 #
 start.time <- Sys.time()
 #knn_lasso_full <-class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range)
 knn_potd <- train(x =x_train_transformed_potd, y =train_data$y, method = "knn", trControl = trainControl(method = "cv", number = 10), preProcess = c("center", "scale"), tuneGrid = expand.grid(k = k_range))
 end.time <- Sys.time()
 time.taken_potd <- round(end.time - start.time,2)
 #knn_lasso_full_prob <- class::knn(train = x_train_transformed_lasso_full, test = x_test_transformed_lasso_full,cl =train_data$y,k=k_range, prob = TRUE)
 
 
 
 knn_time_d10<- c(time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
 names(knn_time_d10)<- c("logistic","lasso", "save","phd","potd")
 print(knn_time_d10)
 
 # plot(rf_full)
 # plot(rf_logistic, add=TRUE)
 # plot(rf_lasso, add=TRUE)
 # rndF1.legend <- if (is.null(rf_full$test$err.rate)) {colnames(rf_full$err.rate)}  else {colnames(rf_full$test$err.rate)}
 # legend("topright", cex =1, legend=rndF1.legend, lty=1, col=c(1,2,3), horiz=T)
 
 #KNNprediction------------------------------------------------------------------------------
 
 predict_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic)
 predict_knn_lasso<- predict(knn_lasso, newdata = x_test_transformed_lasso)
 predict_knn_save<- predict(knn_save, newdata = x_test_transformed_save)
 predict_knn_phd<- predict(knn_phd, newdata = x_test_transformed_phd)
 predict_knn_potd<- predict(knn_potd, newdata = x_test_transformed_potd)
 #Confusion matrix  (KNN)------------------------------------
 conf_knn_logistic<-confusionMatrix(as.factor(predict_knn_logistic),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_lasso<-confusionMatrix(as.factor(predict_knn_lasso),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_save<-confusionMatrix(as.factor(predict_knn_save),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_phd<-confusionMatrix(as.factor(predict_knn_phd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 conf_knn_potd<-confusionMatrix(as.factor(predict_knn_potd),as.factor(test_data[,ncol(test_data)]),mode = "everything")
 #F1 score-----------------------
 # F1_knn<- c(conf_knn_full$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_logistic_full$byClass[7],conf_knn_lasso_full$byClass[7])
 # names(F1_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(F1_knn,3)
 #accuracy_knn-----------------------------
 # accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
 # names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
 # round(accuracy_knn,3)
 #AM risk------------------------------------------
 AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
 AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
 AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
 AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
 AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
 AM_d10<- c(AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
 names(AM_d10)<-  c("logistic","lasso", "save", "phd", "potd")
 round(AM_d10,3)
 ##
 #Missclassification  rate------------------------------------------
 
 MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
 MC_knn_lasso<-1- conf_knn_lasso$overall[1]
 MC_knn_save<-1 - conf_knn_save$overall[1] 
 MC_knn_phd<-1 - conf_knn_phd$overall[1] 
 MC_knn_potd<-1 - conf_knn_potd$overall[1] 
 MC_d10<- c(MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
 names(MC_d10)<-  c("logistic","lasso", "save", "phd", "potd")
 round(MC_d10,3)
 
 
 F1_d10<- c(conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
 names(F1_d10)<- c( "logistic","lasso", "save","phd","potd")
 
 
 ######################################
 ###ROC CURVE------------------------------------------------------------------
 
 #KNN------------------------------------------
 pred_knn_logistic <- predict(knn_logistic, newdata = x_test_transformed_logistic, type = "prob")[, 2]
 pred_knn_lasso <- predict(knn_lasso, newdata = x_test_transformed_lasso, type = "prob")[, 2]
 pred_knn_save <- predict(knn_save, newdata = x_test_transformed_save, type = "prob")[, 2]
 pred_knn_phd <- predict(knn_phd, newdata = x_test_transformed_phd, type = "prob")[, 2]
 pred_knn_potd <- predict(knn_potd, newdata = x_test_transformed_potd, type = "prob")[, 2]
 # Prediction for the ROC------------------------------
 #KNN----------------------------
 pred_knn_logistic <- prediction(pred_knn_logistic, test_data[,ncol(test_data)])
 pred_knn_lasso <- prediction(pred_knn_lasso, test_data[,ncol(test_data)])
 pred_knn_save <- prediction(pred_knn_save, test_data[,ncol(test_data)])
 pred_knn_phd <- prediction(pred_knn_phd, test_data[,ncol(test_data)])
 pred_knn_potd <- prediction(pred_knn_potd, test_data[,ncol(test_data)])
 #Performance----------------------------
 
 #KNN--------------------------
 perf_knn_logistic  <- performance(pred_knn_logistic, "tpr", "fpr")
 perf_knn_lasso  <- performance(pred_knn_lasso, "tpr", "fpr")
 perf_knn_save  <- performance(pred_knn_save, "tpr", "fpr")
 perf_knn_phd  <- performance(pred_knn_phd, "tpr", "fpr")
 perf_knn_potd  <- performance(pred_knn_potd, "tpr", "fpr")
 #Plot------------------------------------
 #ROC
 plot(perf_knn_logistic,colorize = FALSE, col="darkorange", main=expression(paste( d, "= 10")), lty=1, lwd=2)
 rect(par("usr")[1], par("usr")[3],
      par("usr")[2], par("usr")[4],
      col = "#ebebeb")
 
 # Add white grid
 grid(nx = NULL, ny = NULL,
      col = "gray", lwd = 1)
 plot(perf_knn_logistic,colorize = FALSE, add=TRUE,col="darkorange", main="ROC curves for models fitted through knn", lty=1, lwd=2)
 plot(perf_knn_lasso,add=TRUE, colorize = FALSE, col="olivedrab",lty=1,lwd=2)
 plot(perf_knn_save,add=TRUE, colorize = FALSE, col="springgreen", lty=1,lwd=2)
 plot(perf_knn_phd,add=TRUE, colorize = FALSE, col="cyan2",lty=1,lwd=2)
 plot(perf_knn_potd,add=TRUE, colorize = FALSE, col="darkorchid1", lty=1,lwd=2)
 abline(a=0,b=1,lwd=2,lty=2,col="gray")
 # legend("bottomright",legend = c(expression(paste("LLO(", lambda, "= 0)")), 
 #                                 expression(paste("LLO(", lambda, "> 0)")),"SAVE", "PHD" , 
 #                                 "POTD" ),
 #        col = c("darkorange", "olivedrab","springgreen","cyan2","darkorchid1"), lty = 1,lwd=2)
 
 plot(0, type = "n", xlim = c(0, 1), ylim = c(0, 1), xlab = "", ylab = "", main=expression(paste(  "Legends")))
 rect(par("usr")[1], par("usr")[3],
      par("usr")[2], par("usr")[4],
      col = "#ebebeb")
 
 # Add white grid
 grid(nx = NULL, ny = NULL,
      col = "gray", lwd = 1)
 # Add legends
 legend("center",
        legend = c(expression(paste("LLO(", lambda, "= 0)")),
                   expression(paste("LLO(", lambda, "> 0)")),
                   "SAVE", "PHD", "POTD"),
        col = c("darkorange", "olivedrab", "springgreen", "cyan2", "darkorchid1"),
        lty = 1, lwd = 2)
 
 #Area under the curve for knn------------------------------------------------------------------
 AUC_d10<-c( AUC_logsitc=performance(pred_knn_logistic, "auc")@y.values[[1]],
             AUC_lasso=performance(pred_knn_lasso, "auc")@y.values[[1]], AUC_save=performance(pred_knn_save, "auc")@y.values[[1]],AUC_phd=performance(pred_knn_phd, "auc")@y.values[[1]]
             ,AUC_potd=performance(pred_knn_potd, "auc")@y.values[[1]])
 
 AUC_d10
 
 
 
 
 
 
 
 
 
 #Hill_vally data
 n<-length(test_data$y) #for testset
 
 
 library(ggplot2)
 
 
 # Create dataframes for d2, d3, d4, d5, and d10
 df_d2 <- data.frame(d = rep("LLO(lambda = 0)", 5),
                     method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                     x_lower = c(MC_d2[1] - (1.96 * sqrt(MC_d2[1] * (1 - MC_d2[1])) / sqrt(n)), 
                                 MC_d3[1] - (1.96 * sqrt(MC_d3[1] * (1 - MC_d3[1])) / sqrt(n)), 
                                 MC_d4[1] - (1.96 * sqrt(MC_d4[1] * (1 - MC_d4[1])) / sqrt(n)),
                                 MC_d5[1] - (1.96 * sqrt(MC_d5[1] * (1 - MC_d5[1])) / sqrt(n)), 
                                 MC_d10[1] - (1.96 * sqrt(MC_d10[1] * (1 - MC_d10[1])) / sqrt(n))),
                     x = c(MC_d2[1], MC_d3[1], MC_d4[1], MC_d5[1], MC_d10[1]),
                     x_upper = c(MC_d2[1] + (1.96 * sqrt(MC_d2[1] * (1 - MC_d2[1])) / sqrt(n)), 
                                 MC_d3[1] + (1.96 * sqrt(MC_d3[1] * (1 - MC_d3[1])) / sqrt(n)), 
                                 MC_d4[1] + (1.96 * sqrt(MC_d4[1] * (1 - MC_d4[1])) / sqrt(n)),
                                 MC_d5[1] + (1.96 * sqrt(MC_d5[1] * (1 - MC_d5[1])) / sqrt(n)), 
                                 MC_d10[1] + (1.96 * sqrt(MC_d10[1] * (1 - MC_d10[1])) / sqrt(n))))
 
 df_d3 <- data.frame(d = rep("LLO(lambda > 0)", 5),
                     method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                     x_lower = c(MC_d2[2] - (1.96 * sqrt(MC_d2[2] * (1 - MC_d2[2])) / sqrt(n)), 
                                 MC_d3[2] - (1.96 * sqrt(MC_d3[2] * (1 - MC_d3[2])) / sqrt(n)), 
                                 MC_d4[2] - (1.96 * sqrt(MC_d4[2] * (1 - MC_d4[2])) / sqrt(n)),
                                 MC_d5[2] - (1.96 * sqrt(MC_d5[2] * (1 - MC_d5[2])) / sqrt(n)), 
                                 MC_d10[2] - (1.96 * sqrt(MC_d10[2] * (1 - MC_d10[2])) / sqrt(n))),
                     x = c(MC_d2[2], MC_d3[2], MC_d4[2], MC_d5[2], MC_d10[2]),
                     x_upper = c(MC_d2[2] + (1.96 * sqrt(MC_d2[2] * (1 - MC_d2[2])) / sqrt(n)), 
                                 MC_d3[2] + (1.96 * sqrt(MC_d3[2] * (1 - MC_d3[2])) / sqrt(n)), 
                                 MC_d4[2] + (1.96 * sqrt(MC_d4[2] * (1 - MC_d4[2])) / sqrt(n)),
                                 MC_d5[2] + (1.96 * sqrt(MC_d5[2] * (1 - MC_d5[2])) / sqrt(n)), 
                                 MC_d10[2] + (1.96 * sqrt(MC_d10[2] * (1 - MC_d10[2])) / sqrt(n))))
 
 df_d4 <- data.frame(d = rep("SAVE", 5),
                     method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                     x_lower = c(MC_d2[3] - (1.96 * sqrt(MC_d2[3] * (1 - MC_d2[3])) / sqrt(n)), 
                                 MC_d3[3] - (1.96 * sqrt(MC_d3[3] * (1 - MC_d3[3])) / sqrt(n)), 
                                 MC_d4[3] - (1.96 * sqrt(MC_d4[3] * (1 - MC_d4[3])) / sqrt(n)),
                                 MC_d5[3] - (1.96 * sqrt(MC_d5[3] * (1 - MC_d5[3])) / sqrt(n)), 
                                 MC_d10[3] - (1.96 * sqrt(MC_d10[3] * (1 - MC_d10[3])) / sqrt(n))),
                     x = c(MC_d2[3], MC_d3[3], MC_d4[3], MC_d5[3], MC_d10[3]),
                     x_upper = c(MC_d2[3] + (1.96 * sqrt(MC_d2[3] * (1 - MC_d2[3])) / sqrt(n)), 
                                 MC_d3[3] + (1.96 * sqrt(MC_d3[3] * (1 - MC_d3[3])) / sqrt(n)), 
                                 MC_d4[3] + (1.96 * sqrt(MC_d4[3] * (1 - MC_d4[3])) / sqrt(n)),
                                 MC_d5[3] + (1.96 * sqrt(MC_d5[3] * (1 - MC_d5[3])) / sqrt(n)), 
                                 MC_d10[3] + (1.96 * sqrt(MC_d10[3] * (1 - MC_d10[3])) / sqrt(n))))
 
 df_d5 <- data.frame(d = rep("PHD", 5),
                     method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                     x_lower = c(MC_d2[4] - (1.96 * sqrt(MC_d2[4] * (1 - MC_d2[4])) / sqrt(n)), 
                                 MC_d3[4] - (1.96 * sqrt(MC_d3[4] * (1 - MC_d3[4])) / sqrt(n)), 
                                 MC_d4[4] - (1.96 * sqrt(MC_d4[4] * (1 - MC_d4[4])) / sqrt(n)),
                                 MC_d5[4] - (1.96 * sqrt(MC_d5[4] * (1 - MC_d5[4])) / sqrt(n)), 
                                 MC_d10[4] - (1.96 * sqrt(MC_d10[4] * (1 - MC_d10[4])) / sqrt(n))),
                     x = c(MC_d2[4], MC_d3[4], MC_d4[4], MC_d5[4], MC_d10[4]),
                     x_upper = c(MC_d2[4] + (1.96 * sqrt(MC_d2[4] * (1 - MC_d2[4])) / sqrt(n)), 
                                 MC_d3[4] + (1.96 * sqrt(MC_d3[4] * (1 - MC_d3[4])) / sqrt(n)), 
                                 MC_d4[4] + (1.96 * sqrt(MC_d4[4] * (1 - MC_d4[4])) / sqrt(n)),
                                 MC_d5[4] + (1.96 * sqrt(MC_d5[4] * (1 - MC_d5[4])) / sqrt(n)), 
                                 MC_d10[4] + (1.96 * sqrt(MC_d10[4] * (1 - MC_d10[4])) / sqrt(n))))
 
 df_d10 <- data.frame(d = rep("POTD", 5),
                      method = c("d=2", "d=3", "d=4", "d=5", "d=10"),
                      x_lower = c(MC_d2[5] - (1.96 * sqrt(MC_d2[5] * (1 - MC_d2[5])) / sqrt(n)), 
                                  MC_d3[5] - (1.96 * sqrt(MC_d3[5] * (1 - MC_d3[5])) / sqrt(n)), 
                                  MC_d4[5] - (1.96 * sqrt(MC_d4[5] * (1 - MC_d4[5])) / sqrt(n)),
                                  MC_d5[5] - (1.96 * sqrt(MC_d5[5] * (1 - MC_d5[5])) / sqrt(n)), 
                                  MC_d10[5] - (1.96 * sqrt(MC_d10[5] * (1 - MC_d10[5])) / sqrt(n))),
                      x = c(MC_d2[5], MC_d3[5], MC_d4[5], MC_d5[5], MC_d10[5]),
                      x_upper = c(MC_d2[5] + (1.96 * sqrt(MC_d2[5] * (1 - MC_d2[5])) / sqrt(n)), 
                                  MC_d3[5] + (1.96 * sqrt(MC_d3[5] * (1 - MC_d3[5])) / sqrt(n)), 
                                  MC_d4[5] + (1.96 * sqrt(MC_d4[5] * (1 - MC_d4[5])) / sqrt(n)),
                                  MC_d5[5] + (1.96 * sqrt(MC_d5[5] * (1 - MC_d5[5])) / sqrt(n)), 
                                  MC_d10[5] + (1.96 * sqrt(MC_d10[5] * (1 - MC_d10[5])) / sqrt(n))))
 
 # Combine dataframes
 df_combined <- rbind(df_d2, df_d3, df_d4, df_d5, df_d10)
 
 # Define the order of methods
 method_order <- c("d=2", "d=3", "d=4", "d=5", "d=10")
 
 # Convert "method" to ordered factor with custom levels
 df_combined$method <- factor(df_combined$method, levels = method_order, ordered = TRUE)
 
 # Reorder the levels of "d" variable
 df_combined$d <- factor(df_combined$d, levels = c("LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"))
 
 # Plot
 ggplot(df_combined, aes(x = method, y = x, group = d, color = d)) +
   geom_point(position = position_dodge(width = 0.1), size = 2) +
   geom_line(position = position_dodge(width = 0.1), size = 0.5, alpha = 0.7) + # Adjust alpha for line transparency
   geom_errorbar(aes(ymin = x_lower, ymax = x_upper), position = position_dodge(width = 0.1), width = 0.2, alpha = 0.5) +  # Add error bars with adjusted alpha
   labs(x = "Dimensions", y = "Miscl. Risk", title = "") +
   theme(axis.text.x = element_text(angle = 45, hjust = 1),
         legend.position = "top",  # Display legend on top
         legend.key.size = unit(0.5, "lines")) +  # Set smaller legend key size
   scale_color_discrete(name = "Model", breaks = c("LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"))
 
 
 
 
 