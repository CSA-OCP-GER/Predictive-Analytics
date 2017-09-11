## Usage of rxFastForest, rxNeuralNet, rxFastTree and rxFastLinear
## to predict revenue of freeny data set

library(MicrosoftML)

## import data, freeny is used for demo
## rxOdbcData() is used to create DataSet from an ODBC-source.
## For exchanging freeny with your actual dataset, just do dataset <- rxOdbcData()
## getHelp(rxOdbcData) to get help for rxOdbcData
dataset <-rxImport(freeny)
     
## Split data into Training and Testing Set
## Train = 0.7
## Test  = 0.3

dataProb <- c(Train = 0.7, Test = 0.3)
dataSplit <-
  rxSplit(dataset,splitByFactor = "splitVar",transforms = list(splitVar =
  sample(dataFactor,size = .rxNumRows,replace = TRUE,prob = dataProb)),
  transformObjects = list(dataProb = dataProb,dataFactor = factor(names(dataProb),
  levels = names(dataProb))), outFilesBase = tempfile())

## define access to Training and Test set
dataTrain <- dataSplit[[1]]
dataTest <- dataSplit[[2]]

## Formula, prediction for "lag.quarterly.revenue"
## If the dataset has been changed, the formula will need an update
model <- as.formula(lag.quarterly.revenue  ~ market.potential + income.level + price.index) 

## FastTrees 
rxFastTreesFit <- rxFastTrees(model, data=dataTrain, type="regression")
fitTree <- rxPredict(rxFastTreesFit,data=dataTest,suffix=".FastTree")

## FastForest
rxFastForestFit <- rxFastForest(model, data=dataTrain, type="regression")
fitForest <- rxPredict(rxFastForestFit,data=dataTest,suffix=".FastForest")

## NeuralNet
rxNeuralNetFit <- rxNeuralNet(model, data = dataTrain,type="regression")
fitNeuralNet <- rxPredict(rxNeuralNetFit, data=dataTest, suffix = ".NeuralNet")

## FastLinear
rxFastLinearFit <- rxFastLinear(model, data=dataTrain, type="regression")
fitFastLinear <- rxPredict(rxFastLinearFit,data = dataTest, suffix = ".FastLinear")

## Assemble result data frame
predictionResults <- data.frame(fitTree,fitForest,fitNeuralNet,fitFastLinear)

# Use rxDataStep to access data
testData<- rxDataStep(inData = dataTest )
predictionResults <- data.frame(testData,fitTree,fitForest,fitNeuralNet,fitFastLinear)


# Compare Summaries
rxSummary("Score.FastTree ~ lag.quarterly.revenue",predictionResults)
rxSummary("Score.FastForest ~ lag.quarterly.revenue",predictionResults)
rxSummary("Score.NeuralNet ~ lag.quarterly.revenue",predictionResults)
rxSummary("Score.FastLinear ~ lag.quarterly.revenue",predictionResults)

# Plot Results
plot(predictionResults$y, main="Comparison of Regressions",xlim = c(1,15),ylim = c(8,10))
lines(predictionResults$Score.FastForest, col="2", main="FastForest",lty=1)
lines(predictionResults$Score.FastTree,   col="3", main="FastTree",lty=3)
lines(predictionResults$Score.NeuralNet,  col="4", main="NeuralNet",lty=4)
lines(predictionResults$Score.FastLinear,  col="5", main="FastLinear",lty=4)
colSpecs <- c(2,3,4,5)
varNames <- c("FastForest","FastTree","NeuralNet","FastLinear")

legend(x=8,y=9,legend =c("FastForest","FastTree","NeuralNet","FastLinear"),col  = c(2,3,4,5),
       ncol = 1,lwd=1,lty = c(1,3,4,4))