class infor():
    def __init__(self):
        self.modelname = ''
        self.datetime = ''
        self.description = ''
        self.inittype = 'Randomize'
        self.parameters = dict()
        self.trainset = ''
        self.testset = ''
        self.nbclasses = 0
        self.trainsize = 0
        self.testsize = 0
        self.Langevin = 10
        self.LearningRateStartAt = 0.99
        self.LearningRateReduce = 0.2
        self.LearningRateReduceAfter = 1000
        self.LearningRate = 0.5
        self.LangevinRange = {10, 15, 20, 25}
        self.LearningRateRange = {0.5, 0.3, 0.1}
        self.L_Burnin = 0
        self.ParameterChoosing = 'Random search'
        self.Crossvalidation = 'None'
        self.EPOCH = 10
        self.BATCHSIZE = 32
        self.LossFunction = 'CrossEntropyLoss'
        self.TorchNumberThreads = 1
        self.H = {"train_loss": [],"train_acc": []}
        self.OptimalValue = 0
        self.TrainTime = 0.0
        self.TestTime = 0.0
        self.Result = ''
        self.Resultfile = ''
        self.ResultForEpochFile = ''
        self.nSimulate = 1
        self.INIT_LR = 0.001


    def ExportAtEpoch(self, Epo, Los, Acc, Pre, Rec, F1s):
        f = open(self.ResultForEpochFile, mode='a', encoding='utf-8')
        f.write(str(Epo) + ' ' + str(Los) + ' ' + str(Acc) + ' ' + str(Pre) + ' ' + str(Rec) + ' ' + str(F1s) + '\n')
        f.close()

    def Export(self):
        f = open(self.Resultfile, mode='w', encoding='utf-8')
        f.write('Model name: ' + self.modelname + '\n')
        f.write('Date of testing: ' + self.datetime + '\n')
        f.write('Desscription: ' + self.description + '\n')
        f.write('Initial type of starting point: '+ self.inittype + '\n')
        f.write('Algorithm Parameters: \n')
        if self.nSimulate == 0:
            f.write('Simulate Annealing: None')
        else:
            f.write('Simulate Annealing: Yes. Number of Annealing times: ' + str(self.nSimulate))
        for x in self.parameters.keys():
            f.write('    ' + str(x) + ' : ' + str(self.parameters[x]) + '\n')
        f.write('Training set: ' + self.trainset + '\n')
        f.write('Test set: ' + self.testset + '\n')
        f.write('Training size: ' + str(self.trainsize) + '\n')
        f.write('Test size: ' + str(self.testsize) + '\n')
        f.write('Number of classes: ' + str(self.nbclasses) + '\n')
        f.write('Langevin value: ' + str(self.Langevin) + '\n')
        f.write('Langevin range: {')
        for L in self.LangevinRange:
            f.write(str(L) + ', ')
        f.write('}\n')

        f.write('Learning rate start at: ' + str(self.LearningRateStartAt) + '\n')
        f.write('Learning rate reduce: ' + str(self.LearningRateReduce) + '\n')
        f.write('Learning rate reduce after (iterations): ' + str(self.LearningRateReduceAfter) + '\n')
        f.write('Learning rate: ' + str(self.LearningRate) + '\n')
        f.write('Learning rate range: {')
        for Lr in self.LearningRateRange:
            f.write(str(Lr) + ', ')
        f.write('}\n')
        f.write('L_Burnin: ' + str(self.L_Burnin) + '\n')

        f.write('Type of learning rate choosing: ' + self.ParameterChoosing + '\n')
        f.write('Crossvalidation: ' + self.Crossvalidation + '\n')
        f.write('Number of epoch: ' + str(self.EPOCH) + '\n')
        f.write('Batch size     : ' + str(self.BATCHSIZE) + '\n')
        f.write('Loss function  : ' + self.LossFunction + '\n')
        f.write('Number of Torch threads: ' + str(self.TorchNumberThreads) + '\n')

        f.write('The vary of training loss: \n')
        for x in self.H['train_loss']:
            f.write(str(x) + ' ')
        f.write('\nThe results: \n')
        f.write('Optimal objective value: ' + str(self.OptimalValue) + '\n')
        f.write('Traning time: ' + str(self.TrainTime) + '\n')
        f.write('Testing time: ' + str(self.TestTime) + '\n')
        f.write('Result in details : \n')
        f.write('     Precision     Recall      F1-Score     Support\n')
        for x in self.Result.keys():
            f.write(x + '  ')
            if x != 'accuracy':
                for y in self.Result[x].keys():
                    f.write('{0:.6f}     '.format(self.Result[x][y]))
                f.write('\n')
        f.write('Accuracy: ' + str(self.Result['accuracy']))
        f.close()


    def ExportADAM(self):
        f = open(self.Resultfile, mode='w', encoding='utf-8')
        f.write('Model name: ' + self.modelname + '\n')
        f.write('Date of testing: ' + self.datetime + '\n')
        f.write('Desscription: ' + self.description + '\n')
        f.write('Initial type of starting point: '+ self.inittype + '\n')
        f.write('Algorithm Parameters: \n')
        f.write('INITIAL Learning Rate: ' + str(self.INIT_LR) + '\n')
        f.write('Training set: ' + self.trainset + '\n')
        f.write('Test set: ' + self.testset + '\n')
        f.write('Training size: ' + str(self.trainsize) + '\n')
        f.write('Test size: ' + str(self.testsize) + '\n')
        f.write('Number of classes: ' + str(self.nbclasses) + '\n')
        f.write('Crossvalidation: ' + self.Crossvalidation + '\n')
        f.write('Number of epoch: ' + str(self.EPOCH) + '\n')
        f.write('Batch size     : ' + str(self.BATCHSIZE) + '\n')
        f.write('Loss function  : ' + self.LossFunction + '\n')
        f.write('Number of Torch threads: ' + str(self.TorchNumberThreads) + '\n')
        f.write('The vary of training loss: \n')
        for x in self.H['train_loss']:
            f.write(str(x) + ' ')
        f.write('\nThe results: \n')
        f.write('Optimal objective value: ' + str(self.OptimalValue) + '\n')
        f.write('Traning time: ' + str(self.TrainTime) + '\n')
        f.write('Testing time: ' + str(self.TestTime) + '\n')
        f.write('Result in details : \n')
        f.write('     Precision     Recall      F1-Score     Support\n')
        for x in self.Result.keys():
            f.write(x + '  ')
            if x != 'accuracy':
                for y in self.Result[x].keys():
                    f.write('{0:.6f}     '.format(self.Result[x][y]))
                f.write('\n')
        f.write('Accuracy: ' + str(self.Result['accuracy']))
        f.close()


