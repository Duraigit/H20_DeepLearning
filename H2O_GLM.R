# setting the working directory
rm(list=ls(all=TRUE))

# Set working directory
#setwd(choose.dir())

univ=read.table('UnivBank.csv', header=T, sep=',',
                col.names=c('ID','age','exp','inc','zip','family',
                            'ccavg','edu','mortgage','loan',
                            'securities','cd','online','cc'))

# Removing the id, exp, and Zip
univ=univ[,-c(1,3,5)]

# Convert to factors 
univ$family=as.factor(univ$family)
univ$edu=as.factor(univ$edu)
univ$securities=as.factor(univ$securities)
univ$cd=as.factor(univ$cd)
univ$online=as.factor(univ$online)
univ$cc=as.factor(univ$cc)
univ$loan=as.factor(univ$loan)

# Divide the data in to test and train
rows=seq(1,nrow(univ),1)
set.seed(123)
trainRows = sample(rows, nrow(univ)*.8)
set.seed(123)
testRows=rows[-(trainRows)]

train = univ[trainRows,] 
test = univ[testRows,] 

rm(univ, rows, testRows, trainRows)

# Load H2o library
library(h2o)

# Start H2O on the local machine using all available cores and with 4 gigabytes of memory
h2o.init(nthreads = -1, max_mem_size = "1g")

# Import a local R train data frame to the H2O cloud
train.hex <- as.h2o(x = train, destination_frame = "train.hex")

# Lambda search
model_LS = h2o.glm(y = "loan", 
                   x = setdiff(names(train.hex), "loan"),
                   training_frame = train.hex, 
                   family = "binomial",
                   lambda_search = TRUE)

print(model_LS)


# Prepare the parameters for the for H2O glm grid search
lambda_opts = list(list(1), list(.5), list(.1), list(.01), 
                   list(.001), list(.0001), list(.00001), list(0))
alpha_opts = list(list(0), list(.25), list(.5), list(.75), list(1))

hyper_parameters = list(lambda = lambda_opts, alpha = alpha_opts)

# Build H2O GLM with grid search
grid_GLM <- h2o.grid("glm", 
                     hyper_params = hyper_parameters, 
                     grid_id = "grid_GLM.hex",
                     y = "loan", 
                     x = setdiff(names(train.hex), "loan"),
                     training_frame = train.hex, 
                     family = "binomial")

# Remove unused R objects
rm(lambda_opts, alpha_opts, hyper_parameters)

# Get grid summary
summary(grid_GLM)

# Fetch GBM grid models
grid_GLM_models <- lapply(grid_GLM@model_ids, 
                          function(model_id) { h2o.getModel(model_id) })

for (i in 1:length(grid_GLM_models)) 
{ 
  print(sprintf("regularization: %-50s auc: %f", grid_GLM_models[[i]]@model$model_summary$regularization, h2o.auc(grid_GLM_models[[i]])))
}

# Function to find the best model with respective to AUC
find_Best_Model <- function(grid_models){
  best_model = grid_models[[1]]
  best_model_AUC = h2o.auc(best_model)
  for (i in 2:length(grid_models)) 
  {
    temp_model = grid_models[[i]]
    temp_model_AUC = h2o.auc(temp_model)
    if(best_model_AUC < temp_model_AUC)
    {
      best_model = temp_model
      best_model_AUC = temp_model_AUC
    }
  }
  return(best_model)
}

# Find the best model by calling find_Best_Model Function
best_GLM_model = find_Best_Model(grid_GLM_models)

rm(grid_GLM_models)

# Get the auc of the best GBM model
best_GLM_model_AUC = h2o.auc(best_GLM_model)

# Examine the performance of the best model
best_GLM_model

# View the specified parameters of the best model
best_GLM_model@parameters

# Important Variables.
h2o.varimp(best_GLM_model)

# Import a local R test data frame to the H2O cloud
test.hex <- as.h2o(x = test, destination_frame = "test.hex")


# Predict on same training data set
predict.hex = h2o.predict(best_GLM_model, 
                          newdata = test.hex[,setdiff(names(test.hex), "loan")])
       
data_GLM = h2o.cbind(test.hex[,"loan"], predict.hex)
                    
# Copy predictions from H2O to R
pred_GLM = as.data.frame(data_GLM)

# Shutdown H2O
h2o.shutdown(F)

# Hit Rate and Penetration calculation
conf_Matrix_GLM = table(pred_GLM$loan, pred_GLM$predict) 

Accuracy = (conf_Matrix_GLM[1,1]+conf_Matrix_GLM[2,2])/sum(conf_Matrix_GLM)

rm(list=ls())
