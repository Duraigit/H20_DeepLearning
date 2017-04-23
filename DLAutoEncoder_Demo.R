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

# Slit data to train and test
set.seed(1234)
train_index = sample(x = nrow(univ),size = 0.6*nrow(univ))
train = univ[train_index,]
test = univ[-train_index,]

# remove original data set and temporary variables
rm(univ,train_index)

# Build the classification model
model = glm(loan ~ ., data = train,family = 'binomial')

# Predict on train
pred = predict(model,type = "response")
conf_Matrix = table(train$loan,ifelse(pred<0.5,0,1))


#Error Metrics
accuracy_train = sum(diag(conf_Matrix))/sum(conf_Matrix)
precision_train = conf_Matrix[2,2]/sum(conf_Matrix[,2])
recall_train = conf_Matrix[2,2]/sum(conf_Matrix[2,])

# Predict on test
pred = predict(model,newdata = test, type = "response")
conf_Matrix = table(test$loan,ifelse(pred<0.5,0,1))

#Error Metrics
accuracy_test = sum(diag(conf_Matrix))/sum(conf_Matrix)
precision_test = conf_Matrix[2,2]/sum(conf_Matrix[,2])
recall_test = conf_Matrix[2,2]/sum(conf_Matrix[2,])

# remove temp variables
rm(model,pred,conf_Matrix)

#Extract features using autoencoder method
library(h2o)

h2o.init(ip='localhost', port = 54321, max_mem_size = '1g')

# Import a local R train data frame to the H2O cloud
train.hex <- as.h2o(x = train, destination_frame = "train.hex")

y = "loan"
x = setdiff(colnames(train.hex), y)

aec <- h2o.deeplearning(x = x, autoencoder = T, 
                        training_frame=train.hex,
                        activation = "Tanh",
                        hidden = c(20),
                        epochs = 100)

# Extract features from train data
features_train <- as.data.frame(h2o.deepfeatures(data = train.hex[,x], object = aec))

# remove train.hex
h2o.rm(train.hex)

# Import a local R test data frame to the H2O cloud
test.hex <- as.h2o(x = test, destination_frame = "test.hex")

# Extract features from test data
features_test <- as.data.frame(h2o.deepfeatures(data = test.hex[,x], object = aec))

# remove test.hex
h2o.rm(test.hex)

# remove temp variables
rm(x,y,aec)

# add extracted features with original data to train the model
train_new <-data.frame(train,features_train)
test_new <-data.frame(test,features_test)

#remove train,test
rm(train,test)

# Build the classification model using randomForest
require(randomForest)
rf_DL <- randomForest(loan ~ ., data=train_new, keep.forest=TRUE, ntree=30)

# importance of attributes
round(importance(rf_DL), 2)
importanceValues = data.frame(attribute=rownames(round(importance(rf_DL), 2)),MeanDecreaseGini = round(importance(rf_DL), 2))
row.names(importanceValues)=NULL
importanceValues = importanceValues[order(-importanceValues$MeanDecreaseGini),]
# Top 10 Important attributes
Top10ImpAttrs = as.character(importanceValues$attribute[1:16])

Top10ImpAttrs

train_Imp = subset(train_new,select = c(Top10ImpAttrs,"loan"))
test_Imp = subset(test_new,select = c(Top10ImpAttrs,"loan"))

rm(train_new,test_new)
# Build the classification model
model_Imp = glm(loan ~ ., data = train_Imp,family = 'binomial')

# Predict on train
pred = predict(model_Imp,type = "response")
conf_Matrix = table(train_Imp$loan,ifelse(pred<0.5,0,1))


#Error Metrics
accuracy_train_Imp = sum(diag(conf_Matrix))/sum(conf_Matrix)
precision_train_Imp = conf_Matrix[2,2]/sum(conf_Matrix[,2])
recall_train_Imp = conf_Matrix[2,2]/sum(conf_Matrix[2,])

# Predict on test
pred = predict(model_Imp,newdata = test_Imp, type = "response")
conf_Matrix = table(test_Imp$loan,ifelse(pred<0.5,0,1))

#Error Metrics
accuracy_test_Imp = sum(diag(conf_Matrix))/sum(conf_Matrix)
precision_test_Imp = conf_Matrix[2,2]/sum(conf_Matrix[,2])
recall_test_Imp = conf_Matrix[2,2]/sum(conf_Matrix[2,])

rm(pred,conf_Matrix)

accuracy_train
precision_train
recall_train

accuracy_train_Imp
precision_train_Imp
recall_train_Imp

accuracy_test
precision_test
recall_test

accuracy_test_Imp
precision_test_Imp
recall_test_Imp

