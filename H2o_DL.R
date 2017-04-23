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

library(h2o)

h2o.init(ip='localhost', port = 54321, max_mem_size = '1g')

# Import a local R train data frame to the H2O cloud
univ.hex <- as.h2o(x = univ, destination_frame = "univ.hex")

splits <- h2o.splitFrame(univ.hex, 0.6, seed=1234)

y = "loan"
x = setdiff(colnames(univ.hex), y)

dlModel <- h2o.deeplearning(x = x, y = y, 
                            training_frame=splits[[1]],
                            activation = "RectifierWithDropout",
                            hidden = c(20,20),
                            input_dropout_ratio = 0.2,
                            l1 = 1e-5,
                            epochs = 10)

# View specified parameters of the deep learning model
dlModel@parameters

# Examine the performance of the trained model
dlModel # display all performance metrics

# Metrics
h2o.performance(dlModel) 

# Get MSE only
h2o.mse(dlModel)

# Classify the test set (predict class labels)
# This also returns the probability for each class
pred = h2o.predict(dlModel, splits[[2]])

# Take a look at the predictions
head(pred)

# Retrieve the variable importance
h2o.varimp(dlModel)

# Train Deep Learning model and validate on test set and save the variable importances
dlModel <- h2o.deeplearning(x = x, y = y, 
                            training_frame=splits[[1]],
                            activation = "RectifierWithDropout",
                            hidden = c(20,20),
                            input_dropout_ratio = 0.2,
                            l1 = 1e-5, epochs = 10,
                            variable_importances = TRUE)


h2o.varimp(dlModel)

h2o.shutdown(prompt = F)
rm(dlModel,pred,univ.hex)

