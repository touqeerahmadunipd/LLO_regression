###########################
library(dplyr)
library(RANN)
library(glmnet)
#library(GJRM)
library(FactoMineR)
library(caret)
#library(pROC)
library(ROCR)

fit_class<-function(data, sample_size, lambda=NA, weights=NA, k=NA){
  
  coefficient_list <- list()
  df<-data[,-ncol(data)]
  z <- df %>% sample_n(sample_size, replace = FALSE)
  
  for (i in 1:sample_size) {
    query_point <- z[i,]
    k <- k # round(sqrt(NROW(data[, ncol(data)])))  + (round(sqrt(NROW(data[, ncol(data)])))  %% 2 == 0)
    neighbors <- nn2(data[, -ncol(data)], query_point, k = k,  treetype = c("bd"), searchtype = c("standard")) #nearest neigbour search
    nearest_neighbors <- data[neighbors$nn.idx, ]
    
    if ((sum(nearest_neighbors$y == 1) >= 5) & (sum(nearest_neighbors$y == 0) >= 5)) {
      x<- nearest_neighbors[,-ncol(nearest_neighbors)]
      y <- nearest_neighbors[, ncol(nearest_neighbors)]
      #print(y)
      x.new <- scale(x, center = z[i,], scale = FALSE)
      new_x<- as.matrix(x.new) 
      colnames(new_x) <- colnames(x)
      #Weights--------This study will not utilise this step
      w <- ifelse(y == 1, mean(y == 1), 1 - mean(y == 1))   
      
      if (weights==TRUE) {
        suppressWarnings(
          best_model <- glmnet(x.new, y, alpha = 1, family = "binomial",
                               lambda = lambda, weights =w ))
      } else {
        suppressWarnings(best_model <- glmnet(x.new, y, alpha = 1, family = "binomial",
                                              lambda = lambda ))
      }
      
      #)
      coefficients <- coef(best_model)
   
      coefficient_list[[i]] <- coefficients
  
      coefficient_list <- coefficient_list[!sapply(coefficient_list, is.null)]
    } 
    

    
  }
  
  t_par = lapply(coefficient_list, function(x) t(x))

  t_par = do.call(rbind, t_par)
  par_est=t_par[,-1]
  test_mat <- as.matrix(par_est)
  pars <-test_mat
  
  return(pars)
  
  #print(coefficient_list)
  
}









cv.lambda_class_kk<- function(data, weights=NA, k=NA){
  qp <- t(apply(data[,-ncol(data)], 2, mean))
  kk <-k #round(sqrt(length(data[, ncol(data)])))  + (round(sqrt(length(data[, ncol(data)])))  %% 2 == 0)
 
  nb <- nn2(data[, -ncol(data)], qp, k = kk,treetype = c("bd"), searchtype = c("standard")) # search nearest neigbuors
  nbs <- data[nb$nn.idx, ]
 
  if ((sum(nbs$y == 1) >= 1) & (sum(nbs$y == 0) >= 1)) {#if (sum(nbs$y == 1) > 5)
    
    #mean_sub <- scale(nbs[, -ncol(nbs)], center =  qp, scale = FALSE)
    x <- nbs[, -ncol(nbs)]
    y <- nbs[, ncol(nbs)]
    #print(y)
    
    mean_sub <- scale(x, center =  qp, scale = FALSE)
    xx<- as.matrix(mean_sub)
    w <- ifelse(y == 1, mean(y == 1), 1 - mean(y == 1))
    #
    if(weights==TRUE){
      suppressWarnings(cv.fit <- cv.glmnet(xx,y,alpha=1,family="binomial",type.measure = "mse", weights = w))
    }else{
      suppressWarnings(cv.fit <- cv.glmnet(xx,y,alpha=1,family="binomial",type.measure = "mse"))
      
    }
    # plot(cv.fit)
    lambda_min <- cv.fit$lambda.min
    #lambda_min <- cv.fit$lambda.1se
    
    return(lambda_min)
    #print(lambda_min)
  }else
  {
    return(NA)
  }
}



#Dimension selection ---------------

data_fun <- function(data, svd) {
  Vk <- svd$v
  X_transform <- as.matrix(data[,-ncol(data)]) %*% Vk
  d <- data.frame(X_transform, y = data[, ncol(data)])
  train_ind <- sample(1:nrow(d), size = nrow(d) * 0.7, replace = FALSE)
  train_d <- d[train_ind, ]
  train_X <- train_d[, -ncol(train_d)]
  colnames(train_X) <- paste0("x", 1:ncol(train_X))
  train_y <- as.factor(train_d[, ncol(train_d)])
  train_data <- cbind(train_X, y = train_y)
  
  test_d <- d[-train_ind, ]
  test_y <- as.factor(test_d[, ncol(test_d)])
  test_X <- test_d[, -ncol(test_d)]
  colnames(test_X) <- paste0("x", 1:ncol(test_X))
  test_data <- cbind(test_X, y = test_y)
  
  return(list(train_data = train_data, test_data = test_data))
}


# Function for number of component selection---------------------------------------------\
print_progress <- function(iteration, total) {
  percent <- round((iteration / total) * 100, 2)
  cat("\rProgress: ", percent, "%", sep = "")
  flush.console()
}

ncomp_selection3<- function(traindata=NA, testdata=NA,cv =NA,model= c("lasso", "logistic"), method=c( "rf", "knn","lvq")){
  #browser()
  results <- numeric()  # Define the number of trees for the random forest
  #
  train_X<- traindata[,-ncol(traindata)]
  train_y<- traindata[, ncol(traindata)]
  test_X<- testdata[,-ncol(testdata)]
  test_y<- testdata[, ncol(testdata)]
  N<- length(traindata[,-ncol(traindata)])
  for (i in 1:N){
    train_dat<- cbind(train_X[1:i], y=train_y)
    test_dat <- test_X[1:i]
    #print(head(test_dat))
    if(method=="rf"){
      if(cv==FALSE){
        rf_model <- randomForest(y ~ ., data = train_dat, ntree = 500, importance = TRUE, proximity = TRUE)
        predict_y <- predict(rf_model, newdata = test_dat, type = "response", norm.votes = TRUE)
      }else if(cv==TRUE){
        rf_model <- train(y~.,data=train_dat, method='rf', trControl = trainControl(method = "cv", number = 10,, repeats = 3),prox = TRUE)
        predict_y <- predict(object=rf_model,newdata=test_dat)
      }
    }else if(method=="knn"){
      k <- round(sqrt(NROW(train_dat[, ncol(train_dat)])))  + (round(sqrt(NROW(train_dat[, ncol(train_dat)])))  %% 2 == 0)
      knn_model <- train(y ~ ., data = train_dat, method = "knn", trControl = trainControl(method = "repeatedcv", number = 5, repeats = 3),  tuneGrid = expand.grid(k = k))
      predict_y <- predict(knn_model, newdata = test_dat)
    }else if(method=="lvq"){
      lvq_model <- train(y~., data=train_dat, method="lvq", preProcess="scale", trControl=trainControl(method="repeatedcv", number=5, repeats = 3))
      predict_y <- predict(lvq_model, newdata = test_dat)
      
    }
    # ## Print the accuracy
    accuracy <- mean(predict_y == test_y)
    #print(accuracy)
    results[i] <- accuracy
    #print(results)
    
    #print(results)
    print_progress(i, N)
  }
  
  if(method %in%c("rf", "knn", "lvq")){
    #print(results)
    if(model=="logistic"){
      plot(results, type="l", ylab="Accuracy", xlab="Number of components", main=expression(paste(" Components selected via LLO(", lambda, "= 0)")), lwd=2 )
      #For grid---------------------
      rect(par("usr")[1], par("usr")[3],
           par("usr")[2], par("usr")[4],
           col = "#ebebeb")
      # Add white grid
      grid(nx = NULL, ny = NULL,
           col = "white", lwd = 1)
      
      lines(results, type="l", lwd=2 )
      desired_components <- which.max(results)[1]
      abline(v=desired_components, lty=2, col="red")
      return(desired_components)
    } else if(model=="lasso"){
      plot(results, type="l", ylab="Accuracy", xlab="Number of components", main=expression(paste("Components selected via LLO(", lambda, "> 0)")), lwd=2 )
      #For grid---------------------
      rect(par("usr")[1], par("usr")[3],
           par("usr")[2], par("usr")[4],
           col = "#ebebeb")
      # Add white grid
      grid(nx = NULL, ny = NULL,
           col = "white", lwd = 1)
      
      lines(results, type="l" , lwd=2)
      desired_components <- which.max(results)[1]
      abline(v=desired_components, lty=2, col="red")
      return(desired_components)
    }
  }
}

# # Usage
# result <- data_fun(data=data1, svd=svd_logistic)
# set.seed(123)
# result <- data_fun(data=data1, svd=svd_lasso)
# train_data <- result$train_data
# test_data <- result$test_data
# 
# 
# 
#   
# 
# 
# #ncomp_selection(traindata=train_data, testdata=test_data,cv =TRUE,model= c("lasso"), method=c("rf"))




train_test_split = function(X, y, test_size, seed){
  set.seed(seed)
  n=nrow(X)
  test_id = sample(n, round(n*test_size))
  list_final = list("X_train" = X[-test_id,], "X_test" = X[test_id,], 
                    "y_train" = y[-test_id], "y_test" = y[test_id])
  return(list_final)
}





