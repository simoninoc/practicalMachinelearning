Week2 Quiz
#1
library(AppliedPredictiveModeling)
library(caret)
data(AlzheimerDisease)

#2
data(concrete)
library(caret)
set.seed(975)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

#Plot
featurePlot(x = mixtures[,-7], y = mixtures[,7],plot = "pairs")
