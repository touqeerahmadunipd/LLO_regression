#rm(list=ls());gc()
setwd("D:/PostDoc work/Figures/classification/Simulation/code/Github code")
source('Utilities.R')
source('POTD_utility.R')
library(e1071)
library(dr)
#library(Rdimtools)
library(pbapply)
library(class)

train_test_split = function(X, y, test_size, seed){
  set.seed(seed)
  n=nrow(X)
  test_id = sample(n, round(n*test_size))
  list_final = list("X_train" = X[-test_id,], "X_test" = X[test_id,], 
                    "y_train" = y[-test_id], "y_test" = y[test_id])
  return(list_final)
}


generate_binary_response <- function(model, X) {
  epsilon <- rnorm(length(X[,1]), 0, 1)  # Generate random noise
  if (model == "I") {
    Y <- sign(sin(X[,1]) +X[,2]^2 + 0.2 * epsilon)
  } else if (model == "II") {
    Y <- sign((X[,1] + 0.5) * (X[,2] - 0.5)^2 + 0.2 * epsilon)
  } else if (model == "III") {
    #Y <- sign(log(X[,1]^2) * (X[,2]^2 + (X[,3]^2)/2 + (X[,4]^2/4)) + 0.2 * epsilon)
    #Y <- sign(log(X[,1]^2) * ((X[,2]/4) + (X[,3])/2 + (X[,4]^2/4)) + 0.2 * epsilon)
    Y <- sign(log(X[,1]^2) * ((X[,2]^2) + (X[,3]) ) + 0.2 * epsilon)
  } else {
    stop("Invalid model specified.")
  }
  return(Y)
}

#set.seed(123)

#test data-----------
R=1000
n.size=1000
p<-8

X <- matrix(rnorm(p * n.size), ncol = p)
response_I <- generate_binary_response("III", X)
colnames(X) <- paste0("x", 1:p)

test_data<- data.frame(X, y=response_I)

# Example of how to access response data for a specific model
# For example, to access response data for Model I
test_data$y <- ifelse(test_data$y == "-1", 0, test_data$y)
test_data$y<-as.factor(test_data$y)
############


F1.mat_n_500 = matrix(NA,R,7)
AM.mat_n_500 = matrix(NA,R,7)
MC.mat_n_500 = matrix(NA,R,7)
AUC.mat_n_500 = matrix(NA,R,7)
time.mat_n_500 = matrix(NA,R,7)
colnames(F1.mat_n_500) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(AM.mat_n_500) = c( "Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(MC.mat_n_500) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(AUC.mat_n_500) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(time.mat_n_500) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")


##
distance_mat_n_500 = matrix(NA,R,5)
colnames(distance_mat_n_500) = c( "logistic","lasso", "save","phd","potd")




F1.mat_n_1000 = matrix(NA,R,7)
AM.mat_n_1000 = matrix(NA,R,7)
MC.mat_n_1000 = matrix(NA,R,7)
AUC.mat_n_1000 = matrix(NA,R,7)
time.mat_n_1000 = matrix(NA,R,7)
colnames(F1.mat_n_1000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(AM.mat_n_1000) = c( "Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(MC.mat_n_1000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(AUC.mat_n_1000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(time.mat_n_1000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")



##
distance_mat_n_1000 = matrix(NA,R,5)
colnames(distance_mat_n_1000) = c( "logistic","lasso", "save","phd","potd")



F1.mat_n_2000 = matrix(NA,R,7)
AM.mat_n_2000 = matrix(NA,R,7)
MC.mat_n_2000 = matrix(NA,R,7)
AUC.mat_n_2000 = matrix(NA,R,7)
time.mat_n_2000 = matrix(NA,R,7)
colnames(F1.mat_n_2000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(AM.mat_n_2000) = c( "Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(MC.mat_n_2000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(AUC.mat_n_2000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(time.mat_n_2000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")

##
distance_mat_n_2000 = matrix(NA,R,5)
colnames(distance_mat_n_2000) = c( "logistic","lasso", "save","phd","potd")


F1.mat_n_3000 = matrix(NA,R,7)
AM.mat_n_3000 = matrix(NA,R,7)
MC.mat_n_3000 = matrix(NA,R,7)
AUC.mat_n_3000 = matrix(NA,R,7)
time.mat_n_3000 = matrix(NA,R,7)
colnames(F1.mat_n_3000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(AM.mat_n_3000) = c( "Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(MC.mat_n_3000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(AUC.mat_n_3000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(time.mat_n_3000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")


##
distance_mat_n_3000 = matrix(NA,R,5)
colnames(distance_mat_n_3000) = c( "logistic","lasso", "save","phd","potd")



F1.mat_n_4000 = matrix(NA,R,7)
AM.mat_n_4000 = matrix(NA,R,7)
MC.mat_n_4000 = matrix(NA,R,7)
AUC.mat_n_4000 = matrix(NA,R,7)
time.mat_n_4000 = matrix(NA,R,7)
colnames(F1.mat_n_4000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(AM.mat_n_4000) = c( "Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(MC.mat_n_4000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(AUC.mat_n_4000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
colnames(time.mat_n_4000) = c("Full", "Oracle", "logistic","lasso", "save","phd","potd")


##
distance_mat_n_4000 = matrix(NA,R,5)
colnames(distance_mat_n_4000) = c( "logistic","lasso", "save","phd","potd")

lamda= matrix(NA,R,1)
colnames(lamda) = c( "lambda")
###########################################################################
d=3 #-------------------------------------------------------------------
B_true = cbind(c(1,rep(0,p-1)),c(0,1,rep(0,p-2)), c(0,0,1,rep(0,p-3)))
#B_true2 = cbind(c(22,rep(0,p-1)),c(0,2,4,rep(0,p-3)), c(0,0,0,1,rep(0,p-4)))

##########################################################################

n.size <- 500
p<-8
pb <- txtProgressBar(min = 0, max = R, style = 3)
for (r in 1:R) {
  setTxtProgressBar(pb, r)
  # Generate random variables
  X <- matrix(rnorm(p * n.size), ncol = p)
  response_I <- generate_binary_response("III", X)
  colnames(X) <- paste0("x", 1:p)
  
  train_data<- data.frame(X, y=response_I)
  
  # Example of how to access response data for a specific model
  # For example, to access response data for Model I
  train_data$y <- ifelse(train_data$y == "-1", 0, train_data$y)
  train_data$y<-as.factor(train_data$y)
  
  # t=0.95
  # #split train and test-------------------------------
  # train_test_splitt<- train_test_split(X=data[, -ncol(data)], y= data[, ncol(data)], test_size = t, seed = 123)
  # 
  # train_data<- cbind(train_test_splitt$X_train,y= train_data$y ) 
  # test_data<- cbind(train_test_splitt$X_test,y= train_test_splitt$y_test ) 
  # length(train_data$y)
  # length(test_data$y)
  
  coef.mat_logistic<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda = 0, weights = FALSE)
  lambda_min<-cv.lambda_class_kk(data=train_data,weights = FALSE);lambda_min
  coef.mat_lasso<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda =lambda_min , weights = FALSE)
  #
  svd_logistic <- svd(coef.mat_logistic)
  svd_lasso <- svd(coef.mat_lasso)
  
  
  #compititors
  save.fit =dr(train_data$y~.,data=train_data[,-ncol(train_data)], method="save")
  phd.fit = dr(as.numeric(train_data$y)~.,data=train_data[,-ncol(train_data)], method="phdy")
  potd.fit<-potd(X=as.matrix(train_data[,-ncol(train_data)]), y=train_data$y, ndim=ncol(train_data[,-ncol(train_data)]))
  
  
  
  Vk_logistic <- svd_logistic$v[, 1:d]
  Vk_lasso <- svd_lasso$v[, 1:d]
  Vk_save <- save.fit$evectors[, 1:d]
  Vk_phd <- phd.fit$evectors[, 1:d]
  Vk_potd <- potd.fit[, 1:d]
  
  
  
  
  
  distance_logistic = space_dist(B_true, Vk_logistic, type = 2)
  distance_lasso = space_dist(B_true,Vk_lasso, type = 2)
  distance_save = space_dist(B_true, Vk_save, type = 2)
  distance_phd = space_dist(B_true, Vk_phd, type = 2)
  distance_potd = space_dist(B_true,Vk_potd, type = 2)
  distance<- c(distance_logistic,distance_lasso,distance_save,distance_phd,distance_potd)
  names(distance)<- c("logistic","lasso", "save","phd","potd")
  distance_mat_n_500[r,] = distance
  
  ######Transform for logistic and Lasso 
  x_train_transformed_oracle <- as.matrix(train_data[,-ncol(train_data)]) %*% B_true
  x_train_transformed_logistic <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_logistic
  x_train_transformed_lasso <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_lasso
  x_train_transformed_save <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_save
  x_train_transformed_phd <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_phd
  x_train_transformed_potd <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_potd
  
  x_test_transformed_oracle <- as.matrix(test_data[,-ncol(test_data)]) %*% B_true
  x_test_transformed_logistic <- as.matrix(test_data[,-ncol(test_data)]) %*% Vk_logistic
  x_test_transformed_lasso <- as.matrix(test_data[,-ncol(test_data)]) %*% Vk_lasso
  x_test_transformed_save <- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_save
  x_test_transformed_phd<- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_phd
  x_test_transformed_potd<- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_potd
  
  
  
  
  ###KNN-------------------------------------------------------------------
  k_range <- 10#seq(5, 20, by = 1)  # Example range of k values: 1, 3, 5, 7, 9
  #full model------------------------------------------------------------------
  start.time <- Sys.time()
  knn_full<- class::knn(train = train_data[, -ncol(train_data)], test = test_data[,- ncol(test_data)],cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_full <- round(end.time - start.time,2)
  
  
  
  
  #oracle model--------------------------------
  start.time <- Sys.time()
  knn_oracle<- class::knn(train = x_train_transformed_oracle, test = x_test_transformed_oracle,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_oracle<- round(end.time - start.time,2)
  
  #Logistic model--------------------------------
  start.time <- Sys.time()
  knn_logistic<- class::knn(train = x_train_transformed_logistic, test = x_test_transformed_logistic,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_logistic <- round(end.time - start.time,2)
  
  
  
  #Lasso model--------------------------------
  start.time <- Sys.time()
  knn_lasso<- class::knn(train = x_train_transformed_lasso, test = x_test_transformed_lasso,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_lasso <- round(end.time - start.time,2)
  
  
  
  
  #SAVE
  start.time <- Sys.time()
  knn_save <- class::knn(train = x_train_transformed_save, test = x_test_transformed_save,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_save <- round(end.time - start.time,2)
  
  #phD
  start.time <- Sys.time()
  knn_phd <-  class::knn(train = x_train_transformed_phd, test = x_test_transformed_phd,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_phd <- round(end.time - start.time,2)
  
  #potd
  start.time <- Sys.time()
  knn_potd <-  class::knn(train = x_train_transformed_potd, test = x_test_transformed_potd,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_potd<- round(end.time - start.time,2)
  
  
  knn_time<- c(time.taken_full,time.taken_oracle,time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
  names(knn_time)<- c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  time.mat_n_500[r,] = knn_time
  
  
  
  
  
  #KNNprediction------------------------------------------------------------------------------
  conf_knn_full<-confusionMatrix(knn_full, as.factor(test_data$y),mode = "everything")
  conf_knn_oracle<-confusionMatrix(knn_oracle, as.factor(test_data$y),mode = "everything")
  conf_knn_logistic<-confusionMatrix(knn_logistic, as.factor(test_data$y),mode = "everything")
  conf_knn_lasso<-confusionMatrix(knn_lasso, as.factor(test_data$y),mode = "everything")
  conf_knn_save<-confusionMatrix(knn_save, as.factor(test_data$y),mode = "everything")
  conf_knn_phd<-confusionMatrix(knn_phd, as.factor(test_data$y),mode = "everything")
  conf_knn_potd<-confusionMatrix(knn_potd, as.factor(test_data$y),mode = "everything")
  #F1 score-----------------------
  F1<- c(conf_knn_full$byClass[7],conf_knn_oracle$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
  names(F1)<- c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  round(F1,3)
  F1.mat_n_500[r,] = F1
  
  #accuracy_knn-----------------------------
  # accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
  # names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
  # round(accuracy_knn,3)
  #AM risk------------------------------------------
  AM_knn_full<-(1/2)*((1-conf_knn_full$byClass[1])+(1-conf_knn_full$byClass[2]))
  AM_knn_oracle<-(1/2)*((1-conf_knn_oracle$byClass[1])+(1-conf_knn_oracle$byClass[2]))
  AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
  AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
  AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
  AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
  AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
  AM<- c(AM_knn_full,AM_knn_oracle, AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
  names(AM)<-  c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  AM.mat_n_500[r,] = AM
  
  
  
  
  ##
  #Missclassification  rate------------------------------------------
  MC_knn_full<-1 - conf_knn_full$overall[1]
  MC_knn_oracle<-1- conf_knn_oracle$overall[1]
  MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
  MC_knn_lasso<-1- conf_knn_lasso$overall[1]
  MC_knn_save<-1 - conf_knn_save$overall[1] 
  MC_knn_phd<-1 - conf_knn_phd$overall[1]
  MC_knn_potd<-1 - conf_knn_potd$overall[1]
  MC<- c(MC_knn_full,MC_knn_oracle,MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
  names(MC)<-  c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  MC.mat_n_500[r,] = MC
  
}







##---------------------------------------------------------------------------

n.size <- 1000
p<-8




pb <- txtProgressBar(min = 0, max = R, style = 3)
for (r in 1:R) {
  
  setTxtProgressBar(pb, r)
  # Generate random variables
  X <- matrix(rnorm(p * n.size), ncol = p)
  response_I <- generate_binary_response("III", X)
  colnames(X) <- paste0("x", 1:p)
  
  train_data<- data.frame(X, y=response_I)
  
  # Example of how to access response data for a specific model
  # For example, to access response data for Model I
  train_data$y <- ifelse(train_data$y == "-1", 0, train_data$y)
  train_data$y<-as.factor(train_data$y)
  
  # t=0.95
  # #split train and test-------------------------------
  # train_test_splitt<- train_test_split(X=data[, -ncol(data)], y= data[, ncol(data)], test_size = t, seed = 123)
  # 
  # train_data<- cbind(train_test_splitt$X_train,y= train_data$y ) 
  # test_data<- cbind(train_test_splitt$X_test,y= train_test_splitt$y_test ) 
  # length(train_data$y)
  # length(test_data$y)
  
  coef.mat_logistic<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda = 0, weights = FALSE)
  lambda_min<-cv.lambda_class_kk(data=train_data,weights = FALSE);lambda_min
  coef.mat_lasso<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda =lambda_min , weights = FALSE)
  #
  svd_logistic <- svd(coef.mat_logistic)
  svd_lasso <- svd(coef.mat_lasso)
  
  
  #compititors
  save.fit =dr(train_data$y~.,data=train_data[,-ncol(train_data)], method="save")
  phd.fit = dr(as.numeric(train_data$y)~.,data=train_data[,-ncol(train_data)], method="phdy")
  potd.fit<-potd(X=as.matrix(train_data[,-ncol(train_data)]), y=train_data$y, ndim=ncol(train_data[,-ncol(train_data)]))
  
  
  
  Vk_logistic <- svd_logistic$v[, 1:d]
  Vk_lasso <- svd_lasso$v[, 1:d]
  Vk_save <- save.fit$evectors[, 1:d]
  Vk_phd <- phd.fit$evectors[, 1:d]
  Vk_potd <- potd.fit[, 1:d]
  
  
  
  
  
  distance_logistic = space_dist(B_true, Vk_logistic, type = 2)
  distance_lasso = space_dist(B_true,Vk_lasso, type = 2)
  distance_save = space_dist(B_true, Vk_save, type = 2)
  distance_phd = space_dist(B_true, Vk_phd, type = 2)
  distance_potd = space_dist(B_true,Vk_potd, type = 2)
  distance<- c(distance_logistic,distance_lasso,distance_save,distance_phd,distance_potd)
  names(distance)<- c("logistic","lasso", "save","phd","potd")
  distance_mat_n_1000[r,] = distance
  
  ######Transform for logistic and Lasso 
  x_train_transformed_oracle <- as.matrix(train_data[,-ncol(train_data)]) %*% B_true
  x_train_transformed_logistic <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_logistic
  x_train_transformed_lasso <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_lasso
  x_train_transformed_save <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_save
  x_train_transformed_phd <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_phd
  x_train_transformed_potd <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_potd
  
  x_test_transformed_oracle <- as.matrix(test_data[,-ncol(test_data)]) %*% B_true
  x_test_transformed_logistic <- as.matrix(test_data[,-ncol(test_data)]) %*% Vk_logistic
  x_test_transformed_lasso <- as.matrix(test_data[,-ncol(test_data)]) %*% Vk_lasso
  x_test_transformed_save <- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_save
  x_test_transformed_phd<- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_phd
  x_test_transformed_potd<- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_potd
  
  
  
  
  ###KNN-------------------------------------------------------------------
  k_range <- 10#seq(5, 20, by = 1)  # Example range of k values: 1, 3, 5, 7, 9
  #full model------------------------------------------------------------------
  start.time <- Sys.time()
  knn_full<- class::knn(train = train_data[, -ncol(train_data)], test = test_data[,- ncol(test_data)],cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_full <- round(end.time - start.time,2)
  
  
  
  
  #oracle model--------------------------------
  start.time <- Sys.time()
  knn_oracle<- class::knn(train = x_train_transformed_oracle, test = x_test_transformed_oracle,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_oracle<- round(end.time - start.time,2)
  
  #Logistic model--------------------------------
  start.time <- Sys.time()
  knn_logistic<- class::knn(train = x_train_transformed_logistic, test = x_test_transformed_logistic,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_logistic <- round(end.time - start.time,2)
  
  
  
  #Lasso model--------------------------------
  start.time <- Sys.time()
  knn_lasso<- class::knn(train = x_train_transformed_lasso, test = x_test_transformed_lasso,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_lasso <- round(end.time - start.time,2)
  
  
  
  
  #SAVE
  start.time <- Sys.time()
  knn_save <- class::knn(train = x_train_transformed_save, test = x_test_transformed_save,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_save <- round(end.time - start.time,2)
  
  #phD
  start.time <- Sys.time()
  knn_phd <-  class::knn(train = x_train_transformed_phd, test = x_test_transformed_phd,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_phd <- round(end.time - start.time,2)
  
  #potd
  start.time <- Sys.time()
  knn_potd <-  class::knn(train = x_train_transformed_potd, test = x_test_transformed_potd,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_potd<- round(end.time - start.time,2)
  
  
  
  knn_time<- c(time.taken_full,time.taken_oracle,time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
  names(knn_time)<- c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  time.mat_n_1000[r,] = knn_time
  
  
  
  
  
  #KNNprediction------------------------------------------------------------------------------
  conf_knn_full<-confusionMatrix(knn_full, as.factor(test_data$y),mode = "everything")
  conf_knn_oracle<-confusionMatrix(knn_oracle, as.factor(test_data$y),mode = "everything")
  conf_knn_logistic<-confusionMatrix(knn_logistic, as.factor(test_data$y),mode = "everything")
  conf_knn_lasso<-confusionMatrix(knn_lasso, as.factor(test_data$y),mode = "everything")
  conf_knn_save<-confusionMatrix(knn_save, as.factor(test_data$y),mode = "everything")
  conf_knn_phd<-confusionMatrix(knn_phd, as.factor(test_data$y),mode = "everything")
  conf_knn_potd<-confusionMatrix(knn_potd, as.factor(test_data$y),mode = "everything")
  #F1 score-----------------------
  F1<- c(conf_knn_full$byClass[7],conf_knn_oracle$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
  names(F1)<- c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  round(F1,3)
  F1.mat_n_1000[r,] = F1
  
  #accuracy_knn-----------------------------
  # accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
  # names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
  # round(accuracy_knn,3)
  #AM risk------------------------------------------
  AM_knn_full<-(1/2)*((1-conf_knn_full$byClass[1])+(1-conf_knn_full$byClass[2]))
  AM_knn_oracle<-(1/2)*((1-conf_knn_oracle$byClass[1])+(1-conf_knn_oracle$byClass[2]))
  AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
  AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
  AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
  AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
  AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
  AM<- c(AM_knn_full,AM_knn_oracle, AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
  names(AM)<-  c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  AM.mat_n_1000[r,] = AM
  
  
  
  
  ##
  #Missclassification  rate------------------------------------------
  MC_knn_full<-1 - conf_knn_full$overall[1]
  MC_knn_oracle<-1- conf_knn_oracle$overall[1]
  MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
  MC_knn_lasso<-1- conf_knn_lasso$overall[1]
  MC_knn_save<-1 - conf_knn_save$overall[1] 
  MC_knn_phd<-1 - conf_knn_phd$overall[1]
  MC_knn_potd<-1 - conf_knn_potd$overall[1]
  MC<- c(MC_knn_full,MC_knn_oracle,MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
  names(MC)<-  c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  MC.mat_n_1000[r,] = MC
  
}











##---------------------------------------------------------------------------
n.size <- 2000
p<-8




pb <- txtProgressBar(min = 0, max = R, style = 3)
for (r in 1:R) {
  
  setTxtProgressBar(pb, r)
  # Generate random variables
  X <- matrix(rnorm(p * n.size), ncol = p)
  response_I <- generate_binary_response("III", X)
  colnames(X) <- paste0("x", 1:p)
  
  train_data<- data.frame(X, y=response_I)
  
  # Example of how to access response data for a specific model
  # For example, to access response data for Model I
  train_data$y <- ifelse(train_data$y == "-1", 0, train_data$y)
  train_data$y<-as.factor(train_data$y)
  
  # t=0.95
  # #split train and test-------------------------------
  # train_test_splitt<- train_test_split(X=data[, -ncol(data)], y= data[, ncol(data)], test_size = t, seed = 123)
  # 
  # train_data<- cbind(train_test_splitt$X_train,y= train_data$y ) 
  # test_data<- cbind(train_test_splitt$X_test,y= train_test_splitt$y_test ) 
  # length(train_data$y)
  # length(test_data$y)
  coef.mat_logistic<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda = 0, weights = FALSE)
  lambda_min<-cv.lambda_class_kk(data=train_data,weights = FALSE);lambda_min
  coef.mat_lasso<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda =lambda_min , weights = FALSE)
  #
  svd_logistic <- svd(coef.mat_logistic)
  svd_lasso <- svd(coef.mat_lasso)
  
  
  #compititors
  save.fit =dr(train_data$y~.,data=train_data[,-ncol(train_data)], method="save")
  phd.fit = dr(as.numeric(train_data$y)~.,data=train_data[,-ncol(train_data)], method="phdy")
  potd.fit<-potd(X=as.matrix(train_data[,-ncol(train_data)]), y=train_data$y, ndim=ncol(train_data[,-ncol(train_data)]))
  
  
  
  Vk_logistic <- svd_logistic$v[, 1:d]
  Vk_lasso <- svd_lasso$v[, 1:d]
  Vk_save <- save.fit$evectors[, 1:d]
  Vk_phd <- phd.fit$evectors[, 1:d]
  Vk_potd <- potd.fit[, 1:d]
  
  
  
  
  
  distance_logistic = space_dist(B_true, Vk_logistic, type = 2)
  distance_lasso = space_dist(B_true,Vk_lasso, type = 2)
  distance_save = space_dist(B_true, Vk_save, type = 2)
  distance_phd = space_dist(B_true, Vk_phd, type = 2)
  distance_potd = space_dist(B_true,Vk_potd, type = 2)
  distance<- c(distance_logistic,distance_lasso,distance_save,distance_phd,distance_potd)
  names(distance)<- c("logistic","lasso", "save","phd","potd")
  distance_mat_n_2000[r,] = distance
  
  ######Transform for logistic and Lasso 
  x_train_transformed_oracle <- as.matrix(train_data[,-ncol(train_data)]) %*% B_true
  x_train_transformed_logistic <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_logistic
  x_train_transformed_lasso <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_lasso
  x_train_transformed_save <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_save
  x_train_transformed_phd <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_phd
  x_train_transformed_potd <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_potd
  
  x_test_transformed_oracle <- as.matrix(test_data[,-ncol(test_data)]) %*% B_true
  x_test_transformed_logistic <- as.matrix(test_data[,-ncol(test_data)]) %*% Vk_logistic
  x_test_transformed_lasso <- as.matrix(test_data[,-ncol(test_data)]) %*% Vk_lasso
  x_test_transformed_save <- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_save
  x_test_transformed_phd<- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_phd
  x_test_transformed_potd<- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_potd
  
  
  
  
  ###KNN-------------------------------------------------------------------
  k_range <- 10#seq(5, 20, by = 1)  # Example range of k values: 1, 3, 5, 7, 9
  #full model------------------------------------------------------------------
  start.time <- Sys.time()
  knn_full<- class::knn(train = train_data[, -ncol(train_data)], test = test_data[,- ncol(test_data)],cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_full <- round(end.time - start.time,2)
  
  
  
  
  #oracle model--------------------------------
  start.time <- Sys.time()
  knn_oracle<- class::knn(train = x_train_transformed_oracle, test = x_test_transformed_oracle,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_oracle<- round(end.time - start.time,2)
  
  #Logistic model--------------------------------
  start.time <- Sys.time()
  knn_logistic<- class::knn(train = x_train_transformed_logistic, test = x_test_transformed_logistic,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_logistic <- round(end.time - start.time,2)
  
  
  
  #Lasso model--------------------------------
  start.time <- Sys.time()
  knn_lasso<- class::knn(train = x_train_transformed_lasso, test = x_test_transformed_lasso,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_lasso <- round(end.time - start.time,2)
  
  
  
  
  #SAVE
  start.time <- Sys.time()
  knn_save <- class::knn(train = x_train_transformed_save, test = x_test_transformed_save,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_save <- round(end.time - start.time,2)
  
  #phD
  start.time <- Sys.time()
  knn_phd <-  class::knn(train = x_train_transformed_phd, test = x_test_transformed_phd,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_phd <- round(end.time - start.time,2)
  
  #potd
  start.time <- Sys.time()
  knn_potd <-  class::knn(train = x_train_transformed_potd, test = x_test_transformed_potd,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_potd<- round(end.time - start.time,2)
  
  
  
  knn_time<- c(time.taken_full,time.taken_oracle,time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
  names(knn_time)<- c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  time.mat_n_2000[r,] = knn_time
  
  
  
  
  
  #KNNprediction------------------------------------------------------------------------------
  conf_knn_full<-confusionMatrix(knn_full, as.factor(test_data$y),mode = "everything")
  conf_knn_oracle<-confusionMatrix(knn_oracle, as.factor(test_data$y),mode = "everything")
  conf_knn_logistic<-confusionMatrix(knn_logistic, as.factor(test_data$y),mode = "everything")
  conf_knn_lasso<-confusionMatrix(knn_lasso, as.factor(test_data$y),mode = "everything")
  conf_knn_save<-confusionMatrix(knn_save, as.factor(test_data$y),mode = "everything")
  conf_knn_phd<-confusionMatrix(knn_phd, as.factor(test_data$y),mode = "everything")
  conf_knn_potd<-confusionMatrix(knn_potd, as.factor(test_data$y),mode = "everything")
  #F1 score-----------------------
  F1<- c(conf_knn_full$byClass[7],conf_knn_oracle$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
  names(F1)<- c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  round(F1,3)
  F1.mat_n_2000[r,] = F1
  
  #accuracy_knn-----------------------------
  # accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
  # names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
  # round(accuracy_knn,3)
  #AM risk------------------------------------------
  AM_knn_full<-(1/2)*((1-conf_knn_full$byClass[1])+(1-conf_knn_full$byClass[2]))
  AM_knn_oracle<-(1/2)*((1-conf_knn_oracle$byClass[1])+(1-conf_knn_oracle$byClass[2]))
  AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
  AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
  AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
  AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
  AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
  AM<- c(AM_knn_full,AM_knn_oracle, AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
  names(AM)<-  c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  AM.mat_n_2000[r,] = AM
  
  
  
  
  ##
  #Missclassification  rate------------------------------------------
  MC_knn_full<-1 - conf_knn_full$overall[1]
  MC_knn_oracle<-1- conf_knn_oracle$overall[1]
  MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
  MC_knn_lasso<-1- conf_knn_lasso$overall[1]
  MC_knn_save<-1 - conf_knn_save$overall[1] 
  MC_knn_phd<-1 - conf_knn_phd$overall[1]
  MC_knn_potd<-1 - conf_knn_potd$overall[1]
  MC<- c(MC_knn_full,MC_knn_oracle,MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
  names(MC)<-  c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  MC.mat_n_2000[r,] = MC
  
}



##---------------------------------------------------------------------------
n.size <- 3000
p<-8




pb <- txtProgressBar(min = 0, max = R, style = 3)
for (r in 1:R) {
  
  setTxtProgressBar(pb, r)
  # Generate random variables
  X <- matrix(rnorm(p * n.size), ncol = p)
  response_I <- generate_binary_response("III", X)
  colnames(X) <- paste0("x", 1:p)
  
  train_data<- data.frame(X, y=response_I)
  
  # Example of how to access response data for a specific model
  # For example, to access response data for Model I
  train_data$y <- ifelse(train_data$y == "-1", 0, train_data$y)
  train_data$y<-as.factor(train_data$y)
  
  # t=0.95
  # #split train and test-------------------------------
  # train_test_splitt<- train_test_split(X=data[, -ncol(data)], y= data[, ncol(data)], test_size = t, seed = 123)
  # 
  # train_data<- cbind(train_test_splitt$X_train,y= train_data$y ) 
  # test_data<- cbind(train_test_splitt$X_test,y= train_test_splitt$y_test ) 
  # length(train_data$y)
  # length(test_data$y)
  
  coef.mat_logistic<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda = 0, weights = FALSE)
  lambda_min<-cv.lambda_class_kk(data=train_data,weights = FALSE);lambda_min
  coef.mat_lasso<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda =lambda_min , weights = FALSE)
  #
  svd_logistic <- svd(coef.mat_logistic)
  svd_lasso <- svd(coef.mat_lasso)
  
  
  #compititors
  save.fit =dr(train_data$y~.,data=train_data[,-ncol(train_data)], method="save")
  phd.fit = dr(as.numeric(train_data$y)~.,data=train_data[,-ncol(train_data)], method="phdy")
  potd.fit<-potd(X=as.matrix(train_data[,-ncol(train_data)]), y=train_data$y, ndim=ncol(train_data[,-ncol(train_data)]))
  
  
  
  Vk_logistic <- svd_logistic$v[, 1:d]
  Vk_lasso <- svd_lasso$v[, 1:d]
  Vk_save <- save.fit$evectors[, 1:d]
  Vk_phd <- phd.fit$evectors[, 1:d]
  Vk_potd <- potd.fit[, 1:d]
  
  
  
  
  
  distance_logistic = space_dist(B_true, Vk_logistic, type = 2)
  distance_lasso = space_dist(B_true,Vk_lasso, type = 2)
  distance_save = space_dist(B_true, Vk_save, type = 2)
  distance_phd = space_dist(B_true, Vk_phd, type = 2)
  distance_potd = space_dist(B_true,Vk_potd, type = 2)
  distance<- c(distance_logistic,distance_lasso,distance_save,distance_phd,distance_potd)
  names(distance)<- c("logistic","lasso", "save","phd","potd")
  distance_mat_n_3000[r,] = distance
  
  ######Transform for logistic and Lasso 
  x_train_transformed_oracle <- as.matrix(train_data[,-ncol(train_data)]) %*% B_true
  x_train_transformed_logistic <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_logistic
  x_train_transformed_lasso <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_lasso
  x_train_transformed_save <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_save
  x_train_transformed_phd <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_phd
  x_train_transformed_potd <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_potd
  
  x_test_transformed_oracle <- as.matrix(test_data[,-ncol(test_data)]) %*% B_true
  x_test_transformed_logistic <- as.matrix(test_data[,-ncol(test_data)]) %*% Vk_logistic
  x_test_transformed_lasso <- as.matrix(test_data[,-ncol(test_data)]) %*% Vk_lasso
  x_test_transformed_save <- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_save
  x_test_transformed_phd<- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_phd
  x_test_transformed_potd<- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_potd
  
  
  
  
  ###KNN-------------------------------------------------------------------
  k_range <- 10#seq(5, 20, by = 1)  # Example range of k values: 1, 3, 5, 7, 9
  #full model------------------------------------------------------------------
  start.time <- Sys.time()
  knn_full<- class::knn(train = train_data[, -ncol(train_data)], test = test_data[,- ncol(test_data)],cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_full <- round(end.time - start.time,2)
  
  
  
  
  #oracle model--------------------------------
  start.time <- Sys.time()
  knn_oracle<- class::knn(train = x_train_transformed_oracle, test = x_test_transformed_oracle,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_oracle<- round(end.time - start.time,2)
  
  #Logistic model--------------------------------
  start.time <- Sys.time()
  knn_logistic<- class::knn(train = x_train_transformed_logistic, test = x_test_transformed_logistic,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_logistic <- round(end.time - start.time,2)
  
  
  
  #Lasso model--------------------------------
  start.time <- Sys.time()
  knn_lasso<- class::knn(train = x_train_transformed_lasso, test = x_test_transformed_lasso,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_lasso <- round(end.time - start.time,2)
  
  
  
  
  #SAVE
  start.time <- Sys.time()
  knn_save <- class::knn(train = x_train_transformed_save, test = x_test_transformed_save,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_save <- round(end.time - start.time,2)
  
  #phD
  start.time <- Sys.time()
  knn_phd <-  class::knn(train = x_train_transformed_phd, test = x_test_transformed_phd,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_phd <- round(end.time - start.time,2)
  
  #potd
  start.time <- Sys.time()
  knn_potd <-  class::knn(train = x_train_transformed_potd, test = x_test_transformed_potd,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_potd<- round(end.time - start.time,2)
  
  
  
  knn_time<- c(time.taken_full,time.taken_oracle,time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
  names(knn_time)<- c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  time.mat_n_3000[r,] = knn_time
  
  
  
  
  
  #KNNprediction------------------------------------------------------------------------------
  conf_knn_full<-confusionMatrix(knn_full, as.factor(test_data$y),mode = "everything")
  conf_knn_oracle<-confusionMatrix(knn_oracle, as.factor(test_data$y),mode = "everything")
  conf_knn_logistic<-confusionMatrix(knn_logistic, as.factor(test_data$y),mode = "everything")
  conf_knn_lasso<-confusionMatrix(knn_lasso, as.factor(test_data$y),mode = "everything")
  conf_knn_save<-confusionMatrix(knn_save, as.factor(test_data$y),mode = "everything")
  conf_knn_phd<-confusionMatrix(knn_phd, as.factor(test_data$y),mode = "everything")
  conf_knn_potd<-confusionMatrix(knn_potd, as.factor(test_data$y),mode = "everything")
  #F1 score-----------------------
  F1<- c(conf_knn_full$byClass[7],conf_knn_oracle$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
  names(F1)<- c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  round(F1,3)
  F1.mat_n_3000[r,] = F1
  
  #accuracy_knn-----------------------------
  # accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
  # names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
  # round(accuracy_knn,3)
  #AM risk------------------------------------------
  AM_knn_full<-(1/2)*((1-conf_knn_full$byClass[1])+(1-conf_knn_full$byClass[2]))
  AM_knn_oracle<-(1/2)*((1-conf_knn_oracle$byClass[1])+(1-conf_knn_oracle$byClass[2]))
  AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
  AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
  AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
  AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
  AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
  AM<- c(AM_knn_full,AM_knn_oracle, AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
  names(AM)<-  c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  AM.mat_n_3000[r,] = AM
  
  
  
  
  ##
  #Missclassification  rate------------------------------------------
  MC_knn_full<-1 - conf_knn_full$overall[1]
  MC_knn_oracle<-1- conf_knn_oracle$overall[1]
  MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
  MC_knn_lasso<-1- conf_knn_lasso$overall[1]
  MC_knn_save<-1 - conf_knn_save$overall[1] 
  MC_knn_phd<-1 - conf_knn_phd$overall[1]
  MC_knn_potd<-1 - conf_knn_potd$overall[1]
  MC<- c(MC_knn_full,MC_knn_oracle,MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
  names(MC)<-  c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  MC.mat_n_3000[r,] = MC
  
  
}









##---------------------------------------------------------------------------
n.size <- 4000
p<-8




pb <- txtProgressBar(min = 0, max = R, style = 3)
for (r in 1:R) {
  
  setTxtProgressBar(pb, r)
  # Generate random variables
  X <- matrix(rnorm(p * n.size), ncol = p)
  response_I <- generate_binary_response("III", X)
  colnames(X) <- paste0("x", 1:p)
  
  train_data<- data.frame(X, y=response_I)
  
  # Example of how to access response data for a specific model
  # For example, to access response data for Model I
  train_data$y <- ifelse(train_data$y == "-1", 0, train_data$y)
  train_data$y<-as.factor(train_data$y)
  
  # t=0.95
  # #split train and test-------------------------------
  # train_test_splitt<- train_test_split(X=data[, -ncol(data)], y= data[, ncol(data)], test_size = t, seed = 123)
  # 
  # train_data<- cbind(train_test_splitt$X_train,y= train_data$y ) 
  # test_data<- cbind(train_test_splitt$X_test,y= train_test_splitt$y_test ) 
  # length(train_data$y)
  # length(test_data$y)
  
  coef.mat_logistic<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda = 0, weights = FALSE)
  lambda_min<-cv.lambda_class_kk(data=train_data,weights = FALSE);lambda_min
  coef.mat_lasso<-fit_class(data=train_data, sample_size = floor(length(train_data$y)/4), lambda =lambda_min , weights = FALSE)
  #
  svd_logistic <- svd(coef.mat_logistic)
  svd_lasso <- svd(coef.mat_lasso)
  
  
  #compititors
  save.fit =dr(train_data$y~.,data=train_data[,-ncol(train_data)], method="save")
  phd.fit = dr(as.numeric(train_data$y)~.,data=train_data[,-ncol(train_data)], method="phdy")
  potd.fit<-potd(X=as.matrix(train_data[,-ncol(train_data)]), y=train_data$y, ndim=ncol(train_data[,-ncol(train_data)]))
  
  
  
  Vk_logistic <- svd_logistic$v[, 1:d]
  Vk_lasso <- svd_lasso$v[, 1:d]
  Vk_save <- save.fit$evectors[, 1:d]
  Vk_phd <- phd.fit$evectors[, 1:d]
  Vk_potd <- potd.fit[, 1:d]
  
  
  
  
  
  distance_logistic = space_dist(B_true, Vk_logistic, type = 2)
  distance_lasso = space_dist(B_true,Vk_lasso, type = 2)
  distance_save = space_dist(B_true, Vk_save, type = 2)
  distance_phd = space_dist(B_true, Vk_phd, type = 2)
  distance_potd = space_dist(B_true,Vk_potd, type = 2)
  distance<- c(distance_logistic,distance_lasso,distance_save,distance_phd,distance_potd)
  names(distance)<- c("logistic","lasso", "save","phd","potd")
  distance_mat_n_4000[r,] = distance
  
  ######Transform for logistic and Lasso 
  x_train_transformed_oracle <- as.matrix(train_data[,-ncol(train_data)]) %*% B_true
  x_train_transformed_logistic <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_logistic
  x_train_transformed_lasso <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_lasso
  x_train_transformed_save <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_save
  x_train_transformed_phd <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_phd
  x_train_transformed_potd <- as.matrix(train_data[,-ncol(train_data)]) %*% Vk_potd
  
  x_test_transformed_oracle <- as.matrix(test_data[,-ncol(test_data)]) %*% B_true
  x_test_transformed_logistic <- as.matrix(test_data[,-ncol(test_data)]) %*% Vk_logistic
  x_test_transformed_lasso <- as.matrix(test_data[,-ncol(test_data)]) %*% Vk_lasso
  x_test_transformed_save <- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_save
  x_test_transformed_phd<- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_phd
  x_test_transformed_potd<- as.matrix(test_data[,-ncol(test_data)]) %*%Vk_potd
  
  
  
  
  ###KNN-------------------------------------------------------------------
  k_range <- 10#seq(5, 20, by = 1)  # Example range of k values: 1, 3, 5, 7, 9
  #full model------------------------------------------------------------------
  start.time <- Sys.time()
  knn_full<- class::knn(train = train_data[, -ncol(train_data)], test = test_data[,- ncol(test_data)],cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_full <- round(end.time - start.time,2)
  
  
  
  
  #oracle model--------------------------------
  start.time <- Sys.time()
  knn_oracle<- class::knn(train = x_train_transformed_oracle, test = x_test_transformed_oracle,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_oracle<- round(end.time - start.time,2)
  
  #Logistic model--------------------------------
  start.time <- Sys.time()
  knn_logistic<- class::knn(train = x_train_transformed_logistic, test = x_test_transformed_logistic,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_logistic <- round(end.time - start.time,2)
  
  
  
  #Lasso model--------------------------------
  start.time <- Sys.time()
  knn_lasso<- class::knn(train = x_train_transformed_lasso, test = x_test_transformed_lasso,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_lasso <- round(end.time - start.time,2)
  
  
  
  
  #SAVE
  start.time <- Sys.time()
  knn_save <- class::knn(train = x_train_transformed_save, test = x_test_transformed_save,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_save <- round(end.time - start.time,2)
  
  #phD
  start.time <- Sys.time()
  knn_phd <-  class::knn(train = x_train_transformed_phd, test = x_test_transformed_phd,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_phd <- round(end.time - start.time,2)
  
  #potd
  start.time <- Sys.time()
  knn_potd <-  class::knn(train = x_train_transformed_potd, test = x_test_transformed_potd,cl =train_data$y,k=k_range)
  end.time <- Sys.time()
  time.taken_potd<- round(end.time - start.time,2)
  
  
  
  knn_time<- c(time.taken_full,time.taken_oracle,time.taken_logistic,time.taken_lasso,time.taken_save,time.taken_phd,time.taken_potd)
  names(knn_time)<- c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  time.mat_n_4000[r,] = knn_time
  
  
  
  
  
  #KNNprediction------------------------------------------------------------------------------
  conf_knn_full<-confusionMatrix(knn_full, as.factor(test_data$y),mode = "everything")
  conf_knn_oracle<-confusionMatrix(knn_oracle, as.factor(test_data$y),mode = "everything")
  conf_knn_logistic<-confusionMatrix(knn_logistic, as.factor(test_data$y),mode = "everything")
  conf_knn_lasso<-confusionMatrix(knn_lasso, as.factor(test_data$y),mode = "everything")
  conf_knn_save<-confusionMatrix(knn_save, as.factor(test_data$y),mode = "everything")
  conf_knn_phd<-confusionMatrix(knn_phd, as.factor(test_data$y),mode = "everything")
  conf_knn_potd<-confusionMatrix(knn_potd, as.factor(test_data$y),mode = "everything")
  #F1 score-----------------------
  F1<- c(conf_knn_full$byClass[7],conf_knn_oracle$byClass[7],conf_knn_logistic$byClass[7],conf_knn_lasso$byClass[7],conf_knn_save$byClass[7],conf_knn_phd$byClass[7],conf_knn_potd$byClass[7])
  names(F1)<- c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  round(F1,3)
  F1.mat_n_4000[r,] = F1
  
  #accuracy_knn-----------------------------
  # accuracy_knn<- c(conf_knn_full$byClass[11],conf_knn_logistic$byClass[11],conf_knn_lasso$byClass[11],conf_knn_logistic_full$byClass[11],conf_knn_lasso_full$byClass[11])
  # names(accuracy_knn)<- c("Full data model", "DR via Logistic","DR via Lasso","Full component Logistic","Full component Lasso")
  # round(accuracy_knn,3)
  #AM risk------------------------------------------
  AM_knn_full<-(1/2)*((1-conf_knn_full$byClass[1])+(1-conf_knn_full$byClass[2]))
  AM_knn_oracle<-(1/2)*((1-conf_knn_oracle$byClass[1])+(1-conf_knn_oracle$byClass[2]))
  AM_knn_logistic<-(1/2)*((1-conf_knn_logistic$byClass[1])+(1-conf_knn_logistic$byClass[2]))
  AM_knn_lasso<-(1/2)*((1-conf_knn_lasso$byClass[1])+(1-conf_knn_lasso$byClass[2]))
  AM_knn_save<- (1/2)*((1-conf_knn_save$byClass[1])+(1-conf_knn_save$byClass[2]))
  AM_knn_phd<- (1/2)*((1-conf_knn_phd$byClass[1])+(1-conf_knn_phd$byClass[2]))
  AM_knn_potd<- (1/2)*((1-conf_knn_potd$byClass[1])+(1-conf_knn_potd$byClass[2]))
  AM<- c(AM_knn_full,AM_knn_oracle, AM_knn_logistic,AM_knn_lasso, AM_knn_save,AM_knn_phd,AM_knn_potd)
  names(AM)<-  c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  AM.mat_n_4000[r,] = AM
  
  
  
  
  ##
  #Missclassification  rate------------------------------------------
  MC_knn_full<-1 - conf_knn_full$overall[1]
  MC_knn_oracle<-1- conf_knn_oracle$overall[1]
  MC_knn_logistic<-1 - conf_knn_logistic$overall[1]
  MC_knn_lasso<-1- conf_knn_lasso$overall[1]
  MC_knn_save<-1 - conf_knn_save$overall[1] 
  MC_knn_phd<-1 - conf_knn_phd$overall[1]
  MC_knn_potd<-1 - conf_knn_potd$overall[1]
  MC<- c(MC_knn_full,MC_knn_oracle,MC_knn_logistic,MC_knn_lasso, MC_knn_save, MC_knn_phd, MC_knn_potd)
  names(MC)<-  c("Full", "Oracle", "logistic","lasso", "save","phd","potd")
  MC.mat_n_4000[r,] = MC
  
}










#
F1_500<-colMeans(na.omit(F1.mat_n_500))
AM_500<-colMeans(na.omit(AM.mat_n_500))
MC_500<-colMeans(na.omit(MC.mat_n_500))
knn_time.500<-colMeans(na.omit(time.mat_n_500))
#

F1_1000<-colMeans(na.omit(F1.mat_n_1000))
AM_1000<-colMeans(na.omit(AM.mat_n_1000))
MC_1000<-colMeans(na.omit(MC.mat_n_1000))
knn_time.1000<-colMeans(na.omit(time.mat_n_1000))

#
F1_2000<-colMeans(na.omit(F1.mat_n_2000))
AM_2000<-colMeans(na.omit(AM.mat_n_2000))
MC_2000<-colMeans(na.omit(MC.mat_n_2000))
knn_time.2000<-colMeans(na.omit(time.mat_n_2000))
#

F1_3000<-colMeans(na.omit(F1.mat_n_3000))
AM_3000<-colMeans(na.omit(AM.mat_n_3000))
MC_3000<-colMeans(na.omit(MC.mat_n_3000))
knn_time.3000<-colMeans(na.omit(time.mat_n_3000))

#
F1_4000<-colMeans(na.omit(F1.mat_n_4000))
AM_4000<-colMeans(na.omit(AM.mat_n_4000))
MC_4000<-colMeans(na.omit(MC.mat_n_4000))
knn_time.4000<-colMeans(na.omit(time.mat_n_4000))
#



##


##Distance------------
distance_d1<-round(colMeans(na.omit( distance_mat_n_500)),4)
distance_d2<-round(colMeans(na.omit( distance_mat_n_1000)),4)
distance_d3<-round(colMeans(na.omit( distance_mat_n_2000)),4)
distance_d4<-round(colMeans(na.omit( distance_mat_n_3000)),4)
distance_d5<-round(colMeans(na.omit( distance_mat_n_4000)),4)

#distance_d7<-round(colMeans(na.omit( distance_mat_d7)),4)


##



library(ggplot2)

# Create datafrdistancees for d2, d3, d4, d5, and d10
df_d2 <- data.frame(d = rep("LLO(lambda = 0)", 5),
                    method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
                    x = c(distance_d1[1],distance_d2[1], distance_d3[1], distance_d4[1], distance_d5[1]))
df_d3 <- data.frame(d = rep("LLO(lambda > 0)", 5),
                    method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
                    x = c(distance_d1[2],distance_d2[2], distance_d3[2], distance_d4[2], distance_d5[2]))


df_d4 <- data.frame(d = rep("SAVE", 5),
                    method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
                    x = c(distance_d1[3],distance_d2[3], distance_d3[3], distance_d4[3], distance_d5[3]))

df_d5 <- data.frame(d = rep("PHD", 5),
                    method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
                    x = c(distance_d1[4],distance_d2[4], distance_d3[4], distance_d4[4], distance_d5[4]))

df_d10 <- data.frame(d = rep("POTD", 5),
                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
                     x = c(distance_d1[5],distance_d2[5], distance_d3[5], distance_d4[5], distance_d5[5]))

# Combine dataframes
df_combined <- rbind(df_d2,df_d3, df_d4, df_d5, df_d10)

# Define the order of methods
method_order <- c("n=500","n=1000", "n=2000", "n=3000", "n=4000")

# Convert "method" to ordered factor with custom levels
df_combined$method <- factor(df_combined$method, levels = method_order, ordered = TRUE)

# Reorder the levels of "d" variable
df_combined$d <- factor(df_combined$d, levels = c("LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"))

#df_combined$x<-log(df_combined$x)
# Plot
ggplot(df_combined, aes(x = method, y = x, group = d, color = d)) +
  geom_point(position = position_dodge(width = 0.1), size = 3) +
  geom_line(position = position_dodge(width = 0.1), size = 0.5) +
  labs(x = "Sample size", y = "Distance to CS", title = "Example 4") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "top",  # Display legend on top
        legend.key.size = unit(0.5, "lines")) +  # Set smaller legend key size
  scale_color_discrete(name = "Model", breaks = c("LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"),                            labels = c(expression(paste("LLO(", lambda, " = 0)")), expression(paste("LLO(", lambda, " > 0)")), "SAVE", "PHD", "POTD"))




# Create dataframes for d2, d3, d4, d5, and d10
df_d1 <- data.frame(d = rep("Full", 5),
                    method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
                    x = c(MC_500[1],MC_1000[1], MC_2000[1], MC_3000[1], MC_4000[1])) 
df_d2 <- data.frame(d = rep("Oracle", 5),
                    method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
                    x = c(MC_500[2],MC_1000[2], MC_2000[2], MC_3000[2], MC_4000[2])) 


df_d3 <- data.frame(d = rep("LLO(lambda = 0)", 5),
                    method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
                    x = c(MC_500[3],MC_1000[3], MC_2000[3], MC_3000[3], MC_4000[3]))


df_d4 <- data.frame(d = rep("LLO(lambda > 0)", 5),
                    method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
                    x = c(MC_500[4],MC_2000[4], MC_2000[4], MC_3000[4], MC_4000[4]))

df_d5 <- data.frame(d = rep("SAVE", 5),
                    method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
                    x = c(MC_500[5],MC_1000[5], MC_2000[5], MC_3000[5], MC_4000[5]))

df_d6 <- data.frame(d = rep("PHD", 5),
                    method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
                    x = c(MC_500[6],MC_1000[6], MC_2000[6], MC_3000[6], MC_4000[6]))

df_d7 <- data.frame(d = rep("POTD", 5),
                    method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
                    x = c(MC_500[7],MC_1000[7], MC_2000[7], MC_3000[7], MC_4000[7]))

# Combine dataframes
df_MC <- rbind(df_d1, df_d2, df_d3, df_d4, df_d5,df_d6, df_d7)

# Define the order of methods
method_order <- c("n=500","n=1000", "n=2000", "n=3000", "n=4000")

# Convert "method" to ordered factor with custom levels
df_MC$method <- factor(df_MC$method, levels = method_order, ordered = TRUE)

# Reorder the levels of "d" variable
df_MC$d <- factor(df_MC$d, levels = c("Full", "Oracle","LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"))




ggplot(df_MC, aes(x = method, y = x, group = d, color = d, linetype = d)) +
  geom_point(position = position_dodge(width = 0.1), size = 2) +
  geom_line(position = position_dodge(width = 0.1), size = 0.5) +
  labs(x = "Sample size", y = "Mislassification risk", title = "Example 4") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "top",
        legend.key.size = unit(0.1, "lines")) +
  scale_color_manual(name = "Model",
                     values = c("Full" = "red", "Oracle" = "blue", "LLO(lambda = 0)" = "darkorange", 
                                "LLO(lambda > 0)" = "olivedrab", "SAVE" = "springgreen", "PHD" = "cyan2", "POTD" = "darkorchid1"),
                     breaks = c("Full", "Oracle", "LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"),
                     labels = c("Full", "Oracle", expression(paste("LLO(", lambda, " = 0)")), expression(paste("LLO(", lambda, " > 0)")), "SAVE", "PHD", "POTD")) +
  scale_linetype_manual(name = "Model",
                        values = c("Full" = "dashed", "Oracle" = "dashed", "LLO(lambda = 0)" = "solid", 
                                   "LLO(lambda > 0)" = "solid", "SAVE" = "solid", "PHD" = "solid", "POTD" = "solid"),
                        breaks = c("Full", "Oracle", "LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"),
                        labels = c("Full", "Oracle", expression(paste("LLO(", lambda, " = 0)")), expression(paste("LLO(", lambda, " > 0)")), "SAVE", "PHD", "POTD"))


# 
# 
# # Create dataframes for d2, d3, d4, d5, and d10
# df_d1 <- data.frame(d = rep("Full", 5),
#                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
#                     x = c(AM_500[1],AM_1000[1], AM_2000[1], AM_3000[1], AM_4000[1])) 
# df_d2 <- data.frame(d = rep("Oracle", 5),
#                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
#                     x = c(AM_500[2],AM_1000[2], AM_2000[2], AM_3000[2], AM_4000[2])) 
# 
# 
# df_d3 <- data.frame(d = rep("LLO(lambda = 0)", 5),
#                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
#                     x = c(AM_500[3],AM_1000[3], AM_2000[3], AM_3000[3], AM_4000[3]))
# 
# 
# df_d4 <- data.frame(d = rep("LLO(lambda > 0)", 5),
#                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
#                     x = c(AM_500[4],AM_2000[4], AM_2000[4], AM_3000[4], AM_4000[4]))
# 
# df_d5 <- data.frame(d = rep("SAVE", 5),
#                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
#                     x = c(AM_500[5],AM_1000[5], AM_2000[5], AM_3000[5], AM_4000[5]))
# 
# df_d6 <- data.frame(d = rep("PHD", 5),
#                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
#                     x = c(AM_500[6],AM_1000[6], AM_2000[6], AM_3000[6], AM_4000[6]))
# 
# df_d7 <- data.frame(d = rep("POTD", 5),
#                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
#                     x = c(AM_500[7],AM_1000[7], AM_2000[7], AM_3000[7], AM_4000[7]))
# 
# # Combine dataframes
# df_AM <- rbind(df_d1, df_d2, df_d3, df_d4, df_d5,df_d6, df_d7)
# 
# # Define the order of methods
# method_order <- c("n=500","n=1000", "n=2000", "n=3000", "n=4000")
# 
# # Convert "method" to ordered factor with custom levels
# df_AM$method <- factor(df_AM$method, levels = method_order, ordered = TRUE)
# 
# # Reorder the levels of "d" variable
# df_AM$d <- factor(df_AM$d, levels = c("Full", "Oracle","LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"))
# 
# setwd("D:/PostDoc work/Figures/classification/Simulation/code/Ex4")
# save(df_AM, file = "AM_ex4.Rdata")
# # Plot
# 
# # Plot
# 
# 
# ggplot(df_AM, aes(x = method, y = x, group = d, color = d, linetype = d)) +
#   geom_point(position = position_dodge(width = 0.1), size = 2) +
#   geom_line(position = position_dodge(width = 0.1), size = 0.5) +
#   labs(x = "Sample size", y = "AM risk", title = "") +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1),
#         legend.position = "top",
#         legend.key.size = unit(0.1, "lines")) +
#   scale_color_manual(name = "Model",
#                      values = c("Full" = "red", "Oracle" = "blue", "LLO(lambda = 0)" = "darkorange", 
#                                 "LLO(lambda > 0)" = "olivedrab", "SAVE" = "springgreen", "PHD" = "cyan2", "POTD" = "darkorchid1"),
#                      breaks = c("Full", "Oracle", "LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"),
#                      labels = c("Full", "Oracle", expression(paste("LLO(", lambda, " = 0)")), expression(paste("LLO(", lambda, " > 0)")), "SAVE", "PHD", "POTD")) +
#   scale_linetype_manual(name = "Model",
#                         values = c("Full" = "dashed", "Oracle" = "dashed", "LLO(lambda = 0)" = "solid", 
#                                    "LLO(lambda > 0)" = "solid", "SAVE" = "solid", "PHD" = "solid", "POTD" = "solid"),
#                         breaks = c("Full", "Oracle", "LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"),
#                         labels = c("Full", "Oracle", expression(paste("LLO(", lambda, " = 0)")), expression(paste("LLO(", lambda, " > 0)")), "SAVE", "PHD", "POTD"))
# 
# 
# 
# 
# 
# ###-----------------------------------
# # Create dataframes for d2, d3, d4, d5, and d10
# df_d1 <- data.frame(d = rep("Full", 5),
#                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
#                     x = c(F1_500[1],F1_1000[1], F1_2000[1], F1_3000[1], F1_4000[1])) 
# df_d2 <- data.frame(d = rep("Oracle", 5),
#                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
#                     x = c(F1_500[2],F1_1000[2], F1_2000[2], F1_3000[2], F1_4000[2])) 
# 
# 
# df_d3 <- data.frame(d = rep("LLO(lambda = 0)", 5),
#                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
#                     x = c(F1_500[3],F1_1000[3], F1_2000[3], F1_3000[3], F1_4000[3]))
# 
# 
# df_d4 <- data.frame(d = rep("LLO(lambda > 0)", 5),
#                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
#                     x = c(F1_500[4],F1_2000[4], F1_2000[4], F1_3000[4], F1_4000[4]))
# 
# df_d5 <- data.frame(d = rep("SAVE", 5),
#                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
#                     x = c(F1_500[5],F1_1000[5], F1_2000[5], F1_3000[5], F1_4000[5]))
# 
# df_d6 <- data.frame(d = rep("PHD", 5),
#                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
#                     x = c(F1_500[6],F1_1000[6], F1_2000[6], F1_3000[6], F1_4000[6]))
# 
# df_d7 <- data.frame(d = rep("POTD", 5),
#                     method = c("n=500","n=1000", "n=2000", "n=3000", "n=4000"),
#                     x = c(F1_500[7],F1_1000[7], F1_2000[7], F1_3000[7], F1_4000[7]))
# 
# # Combine dataframes
# df_F1 <- rbind(df_d1, df_d2, df_d3, df_d4, df_d5,df_d6, df_d7)
# 
# # Define the order of methods
# method_order <- c("n=500","n=1000", "n=2000", "n=3000", "n=4000")
# 
# # Convert "method" to ordered factor with custom levels
# df_F1$method <- factor(df_F1$method, levels = method_order, ordered = TRUE)
# 
# # Reorder the levels of "d" variable
# df_F1$d <- factor(df_F1$d, levels = c("Full", "Oracle","LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"))
# 
# 
# save(df_F1, file = "F1_ex4.Rdata")
# # Plot
# 
# 
# 
# ggplot(df_F1, aes(x = method, y = x, group = d, color = d, linetype = d)) +
#   geom_point(position = position_dodge(width = 0.1), size = 2) +
#   geom_line(position = position_dodge(width = 0.1), size = 0.5) +
#   labs(x = "Sample size", y = "F1 score", title = "") +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1),
#         legend.position = "top",
#         legend.key.size = unit(0.1, "lines")) +
#   scale_color_manual(name = "Model",
#                      values = c("Full" = "red", "Oracle" = "blue", "LLO(lambda = 0)" = "darkorange", 
#                                 "LLO(lambda > 0)" = "olivedrab", "SAVE" = "springgreen", "PHD" = "cyan2", "POTD" = "darkorchid1"),
#                      breaks = c("Full", "Oracle", "LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"),
#                      labels = c("Full", "Oracle", expression(paste("LLO(", lambda, " = 0)")), expression(paste("LLO(", lambda, " > 0)")), "SAVE", "PHD", "POTD")) +
#   scale_linetype_manual(name = "Model",
#                         values = c("Full" = "dashed", "Oracle" = "dashed", "LLO(lambda = 0)" = "solid", 
#                                    "LLO(lambda > 0)" = "solid", "SAVE" = "solid", "PHD" = "solid", "POTD" = "solid"),
#                         breaks = c("Full", "Oracle", "LLO(lambda = 0)", "LLO(lambda > 0)", "SAVE", "PHD", "POTD"),
#                         labels = c("Full", "Oracle", expression(paste("LLO(", lambda, " = 0)")), expression(paste("LLO(", lambda, " > 0)")), "SAVE", "PHD", "POTD"))
# 
