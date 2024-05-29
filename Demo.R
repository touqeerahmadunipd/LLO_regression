#rm(list=ls());gc()
setwd("D:/PostDoc work/Figures/classification/Simulation/code/Github code")
source('Utilities.R')
source('POTD_utility.R')
library(e1071)
library(dr)
#library(Rdimtools)
library(pbapply)
library(class)

##Function for generating binary response-------
generate_binary_response <- function(model, X) {
  epsilon <- rnorm(length(X[,1]), 0, 1)  # Generate random noise
  if (model == "I") {
    Y <- sign(sin(X[,1]) +X[,2]^2 + 0.2 * epsilon)
  } else if (model == "II") {
    Y <- sign((X[,1] + 0.5) * (X[,2] - 0.5)^2 + 0.2 * epsilon)
  } else if (model == "III") {
    Y <- sign(log(X[,1]^2) * ((X[,2]^2) + (X[,3]) ) + 0.2 * epsilon)
  } else {
    stop("Invalid model specified.")
  }
  return(Y)
}

#data-----------------------------
n.size=1000
p<-8

X <- matrix(rnorm(p * n.size), ncol = p)
response_I <- generate_binary_response("III", X)
colnames(X) <- paste0("x", 1:p)
data<- data.frame(X, y=response_I)
data$y <- ifelse(data$y == "-1", 0, data$y) #made y in 0, 1
data$y<-as.factor(data$y)
############

k=round(sqrt(NROW(data[, ncol(data)])))  + (round(sqrt(NROW(data[, ncol(data)])))  %% 2 == 0) # k=sqrt(n) nearest neighbors
fit.logistic<-fit_class(data=data, sample_size = floor(length(data$y)/4), lambda = 0, weights = FALSE,k=k )
lambda_min<-cv.lambda_class_kk(data=data,weights = FALSE, k=k);lambda_min
fit.lasso<-fit_class(data=data, sample_size = floor(length(data$y)/4), lambda =lambda_min , weights = FALSE, k=k)
#
svd_logistic <- svd(fit.logistic)
svd_lasso <- svd(fit.lasso)
  
  
#competitors---------
save.fit =dr(data$y~.,data=data[,-ncol(data)], method="save")
phd.fit = dr(as.numeric(data$y)~.,data=data[,-ncol(data)], method="phdy")
potd.fit<-potd(X=as.matrix(data[,-ncol(data)]), y=data$y, ndim=ncol(data[,-ncol(data)]))
  
###########################################################################
d=3 #True dimensions-------------------------------------------------------------------
B_true = cbind(c(1,rep(0,p-1)),c(0,1,rep(0,p-2)), c(0,0,1,rep(0,p-3))) #True central subspace
##
Vk_logistic <- svd_logistic$v[, 1:d]
Vk_lasso <- svd_lasso$v[, 1:d]
Vk_save <- save.fit$evectors[, 1:d]
Vk_phd <- phd.fit$evectors[, 1:d]
Vk_potd <- potd.fit[, 1:d]
  
##Distance to central subspace
distance_logistic = space_dist(B_true, Vk_logistic, type = 2)
distance_lasso = space_dist(B_true,Vk_lasso, type = 2)
distance_save = space_dist(B_true, Vk_save, type = 2)
distance_phd = space_dist(B_true, Vk_phd, type = 2)
distance_potd = space_dist(B_true,Vk_potd, type = 2)
distance<- c(distance_logistic,distance_lasso,distance_save,distance_phd,distance_potd)
names(distance)<- c("logistic","lasso", "save","phd","potd")
distance
  