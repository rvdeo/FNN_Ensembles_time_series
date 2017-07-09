# Rohitash Chandra, 2017 c.rohitash@gmail.conm

#!/usr/bin/python

# built using: https://github.com/rohitash-chandra/VanillaFNN-Python
 

#Sigmoid units used in hidden and output layer. gradient descent is used 
  

#Enemble learning for feedforward neural networks. 
 
 
import numpy as np 
import random
import time
import csv
import matplotlib.pyplot as plt

#An example of a class
class Network:

    def __init__(self, Topo, Train, Test, MaxTime,  MinPer): 
        self.Top  = Topo  # NN topology [input, hidden, output]
        self.Max = MaxTime # max epocs
        self.TrainData = Train
        self.TestData = Test
        self.NumSamples = Train.shape[0]

        self.lrate  = 0 # will be updated later with BP call

        self.momenRate = 0
        self.useNesterovMomen = 0 #use nestmomentum 1, not use is 0

        self.minPerf = MinPer
                                        #initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
    	np.random.seed() 
    	self.W1 = np.random.randn(self.Top[0]  , self.Top[1])  / np.sqrt(self.Top[0] ) 
        self.B1 = np.random.randn(1  , self.Top[1])  / np.sqrt(self.Top[1] ) # bias first layer
        self.BestB1 = self.B1
        self.BestW1 = self.W1 
    	self.W2 = np.random.randn(self.Top[1] , self.Top[2]) / np.sqrt(self.Top[1] )
        self.B2 = np.random.randn(1  , self.Top[2])  / np.sqrt(self.Top[1] ) # bias second layer
        self.BestB2 = self.B2
        self.BestW2 = self.W2 
        self.hidout = np.zeros((1, self.Top[1] )) # output of first hidden layer
        self.out = np.zeros((1, self.Top[2])) #  output last layer

  
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def printNet(self):
        print self.Top
        print self.W1

    def sampleEr(self,actualout):
        error = np.subtract(self.out, actualout)
        sqerror= np.sum(np.square(error))/self.Top[2]
        #print sqerror
        return sqerror
  
    def ForwardPass(self, X ): 
         z1 = X.dot(self.W1) - self.B1  
         self.hidout = self.sigmoid(z1) # output of first hidden layer   
         z2 = self.hidout.dot(self.W2)  - self.B2 
         self.out = self.sigmoid(z2)  # output second hidden layer
  	 
 
  
    def BackwardPassMomentum(self, Input, desired, vanilla):   
            out_delta =   (desired - self.out)*(self.out*(1-self.out))  
            hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1-self.hidout))
            
            if vanilla == 1: #no momentum 
                self.W2+= (self.hidout.T.dot(out_delta) * self.lrate)  
                self.B2+=  (-1 * self.lrate * out_delta)
                self.W1 += (Input.T.dot(hid_delta) * self.lrate) 
                self.B1+=  (-1 * self.lrate * hid_delta)
              
            else:
 	                  # momentum http://cs231n.github.io/neural-networks-3/#sgd 
            	self.W2 += ( self.W2 *self.momenRate) + (self.hidout.T.dot(out_delta) * self.lrate)       # velocity update
            	self.W1 += ( self.W1 *self.momenRate) + (Input.T.dot(hid_delta) * self.lrate)   
                self.B2 += ( self.B2 *self.momenRate) + (-1 * self.lrate * out_delta)       # velocity update
            	self.B1 += ( self.B1 *self.momenRate) + (-1 * self.lrate * hid_delta)   

          

    def Predict(self, Data):
		Input = np.zeros((1, self.Top[0])) # temp hold input
		Desired = np.zeros((1, self.Top[2])) 
		nOutput = np.zeros((1, self.Top[2]))
		#testSize = Data.shape[0]
		self.W1 = self.BestW1
		self.W2 = self.BestW2 #load best knowledge
		self.B1 = self.BestB1
		self.B2 = self.BestB2 #load best knowledge

		#for s in xrange(0, Data.):      
		Input[:]  =   Data[0:self.Top[0]] 
		Desired[:] =  Data[self.Top[0]:] 
		self.ForwardPass(Input )
		#print self.out
		#raw_input("Press Enter to continue...")
			
		return (self.out[0:self.Top[2]] )
		
			       
    
    def TestNetwork(self, Data,  erTolerance):
        Input = np.zeros((1, self.Top[0])) # temp hold input
        Desired = np.zeros((1, self.Top[2])) 
        nOutput = np.zeros((1, self.Top[2]))
        clasPerf = 0
     	sse = 0  
        testSize = Data.shape[0]
        #print "test size = ", testSize
        self.W1 = self.BestW1
        self.W2 = self.BestW2 #load best knowledge
        self.B1 = self.BestB1
        self.B2 = self.BestB2 #load best knowledge
     
        for s in xrange(0, testSize):  
                Input[:]  =   Data[s,0:self.Top[0]] 
                Desired[:] =  Data[s,self.Top[0]:] 
               
                self.ForwardPass(Input ) 
                sse = sse+ self.sampleEr(Desired)  
		 
                if(np.isclose(self.out, Desired, atol=erTolerance).any()):
                   clasPerf =  clasPerf +1                                  
	
	rmse = np.sqrt(sse/testSize*self.Top[2])
	
	return ( rmse, float(clasPerf)/testSize * 100 )

	

 
    def saveKnowledge(self):
        self.BestW1 = self.W1
        self.BestW2 = self.W2
        self.BestB1 = self.B1
        self.BestB2 = self.B2  
 
    def BP_GD(self, learnRate, mRate,    stocastic, vanilla): # BP with SGD (Stocastic BP)
        self.lrate = learnRate
        self.momenRate = mRate 
     
        Input = np.zeros((1, self.Top[0])) # temp hold input
        Desired = np.zeros((1, self.Top[2])) 
        #Er = []#np.zeros((1, self.Max)) 
        epoch = 0
        bestmse = 100
        bestTrain = 0
        rmse = 0
        while  epoch < self.Max :#and bestTrain < self.minPerf :
           
            sse = 0
            for pat in xrange(0, self.NumSamples):
          	Input[:]  =  self.TrainData[pat,0:self.Top[0]]  
                Desired[:] = self.TrainData[pat,self.Top[0]:]  
                self.ForwardPass(Input )  
                self.BackwardPassMomentum(Input , Desired, vanilla)
                sse = sse+ self.sampleEr(Desired)
             
            rmse = np.sqrt(sse/self.NumSamples*self.Top[2])

            if rmse < bestmse:
               bestmse = rmse
               self.saveKnowledge() 
               (x,bestTrain) = self.TestNetwork(self.TrainData,   0.2)
              

            #Er = np.append(Er, rmse)
 

            epoch=epoch+1  

        return (rmse,bestmse, bestTrain, epoch) 
    
	

#--------------------------------------------------------------------------------------------------------





class MTnetwork: # Multi-Task leaning using Stocastic GD 

    def __init__(self, mtaskNet, trainData, testData, maxTime, minPerf, learnRate, numModules, transKnow):
          #trainData and testData could also be different datasets. this example considers one dataset
	self.transKnowlege = transKnow
	self.trainData = trainData
	self.testData = testData
	self.maxTime = maxTime
	self.minCriteria = minPerf
        self.numModules = numModules # number of network modules (i.e tasks with shared knowledge representation)
                           # need to define network toplogies for the different tasks. 
        
        self.mtaskNet = mtaskNet 

        self.learnRate = learnRate
        self.trainTolerance = 0.20 # [eg 0.15 output would be seen as 0] [ 0.81 would be seen as 1]
        self.testTolerance = 0.49
         

    def transferKnowledge(self, Wprev, Wnext): # transfer knowledge (weights from given layer) from  Task n (Wprev) to Task n+1 (Wnext) 
        x=0
        y = 0 
        Wnext[x:x+Wprev.shape[0], y:y+Wprev.shape[1]] = Wprev                                   #(Netlist[n].W1 ->  Netlist[n+1].W1)
        return Wnext
        

    def Procedure(self): 
          
        mRate = 0.05
       
        stocastic = 1 # 0 for vanilla BP. 1 for Stocastic BP
        vanilla = 1 # 1 for Vanilla Gradient Descent, 0 for Gradient Descent with momentum
      

        Netlist = [None]*20  # create list of Network objects ( just max size of 10 for now )
        
        depthSearch = 1

        trainPerf = np.zeros(self.numModules)
        trainMSE =  np.zeros(self.numModules)
        testPerf = np.zeros(self.numModules)
        testMSE =  np.zeros(self.numModules)
	
	
	
        erPlot = np.random.randn((self.maxTime/(depthSearch)*self.numModules),self.numModules)  
         # plot of convergence for each module (Netlist[n] )
       
	
	trdat = splitData(self.trainData, (self.numModules))
	tsdat = splitData(self.testData, (self.numModules))
	

        for n in xrange(0, self.numModules): 
			Netlist[n] = Network(self.mtaskNet[n], trdat[n], tsdat[n], depthSearch,  self.minCriteria) 
			
	
        cycles = 0     
        index = 0
        
        
        while(depthSearch*cycles) <(self.maxTime*self.numModules):
		cycles =cycles + 1
        	for n in xrange(0, self.numModules):  
            		(erPlot[index, n],  trainMSE[n], trainPerf[n], Epochs) = Netlist[n].BP_GD(self.learnRate, mRate,   stocastic, vanilla) 
		index = index + 1
				

	print trainMSE
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# create stacked training dataset
#-------------------------------------------------------------------------------------------
	inputNeurons = 	self.mtaskNet[0][0]
	outputNeurons = self.mtaskNet[0][2]
	
        stacked_dataset= list()
        for row in self.trainData:
			stacked_row = list()
			#for i in xrange(0, inputNeurons ): #use if you want to concatenate original data into stacked file
			#	stacked_row.append(row[i])
				
			for n in xrange(0, self.numModules):				
				prediction = Netlist[n].Predict(row)[0]
				for predict in prediction:
					stacked_row.append(predict)
			for val in xrange(inputNeurons, inputNeurons+outputNeurons):
				stacked_row.append(row[val])
			
			
			stacked_dataset.append(stacked_row)
			#print stacked_dataset
			#raw_input("Press Enter to continue...")
			
        with open('myfile.csv','w') as f:
		   	writer = csv.writer(f)
		   	writer.writerows(stacked_dataset)
	
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# create NN to predict based on stacked data And train on stacked data
#-------------------------------------------------------------------------------------------
	
       	comb_struc = [(self.numModules*outputNeurons) ,self.mtaskNet[0][1],outputNeurons]	
       	
       	tr_st_dt  = np.loadtxt("myfile.csv", delimiter=',') 
        st_dt_tr  = normalisedata(tr_st_dt, comb_struc[0], comb_struc[2])
       		
       	NNCombiner=  Network(comb_struc, tr_st_dt, self.testData, depthSearch,  self.minCriteria)
       	
	cycles = 0     
        errorPlot=  np.random.randn(self.maxTime/(depthSearch),1)
        evaluations = depthSearch
        
        while(evaluations <= self.maxTime):

       		(errorPlot[cycles],  trMSE, trPerf, Epochs) = NNCombiner.BP_GD(self.learnRate, mRate,   stocastic, vanilla) 
       		evaluations = depthSearch + evaluations
       		cycles =cycles + 1
       	
       
        
        #plt.figure()
	#plt.plot(erPlot) # plots all the net modules from the last exp run
	#plt.plot(errorPlot)
	#plt.ylabel('error')  
	#plt.savefig('out.png') 
	
	


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# Test ensamble network  
#-------------------------------------------------------------------------------------------
	#create testdata -> stacked data
        stacked_dataset= list()
        for row in self.testData:
			stacked_row = list()

			#for i in xrange(0, inputNeurons ):  #use if you want to concatenate original data into stacked file
			#	stacked_row.append(row[i])
				
			for n in xrange(0, self.numModules):				
				prediction = Netlist[n].Predict(row)[0]
				for predict in prediction:
					stacked_row.append(predict)
			for val in xrange(inputNeurons, inputNeurons+outputNeurons):
				stacked_row.append(row[val])
			
			stacked_dataset.append(stacked_row)
         
        with open('stack_data.csv','w') as f:
		   	writer = csv.writer(f)
		   	writer.writerows(stacked_dataset)

	#test on combiner network
        st_dt = np.loadtxt("stack_data.csv", delimiter=',')
      
        
        #test on both training ad testing data again
        (trMSE, trPerf) = NNCombiner.TestNetwork(tr_st_dt, self.trainTolerance) 
        (tesMSE, tesPerf) = NNCombiner.TestNetwork(st_dt,  self.testTolerance) 
        #print trMSE
        #print tesMSE
   	
     
        return ( trMSE,  tesMSE, trPerf, tesPerf)
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------


	
def normalisedata(data, inputsize, outsize): # normalise the data between [0,1]
    traindt = data[:,np.array(range(0,inputsize))]	
    dt = np.amax(traindt, axis=0)
    #tds = abs(traindt/dt) 
    tds = traindt
    return np.concatenate(( tds[:,range(0,inputsize)], data[:,range(inputsize,inputsize+outsize)]), axis=1)




# ------------------------------------------------------------------------------------------------------

def splitData( dat, partitions):
	size = len(dat)/partitions
	lensize=size
	split_data = list()
	counter = 0

	for x in xrange(0,partitions-1):
		listdat = dat[counter:lensize]
		counter=counter+size
		lensize = lensize+size
		split_data.append(listdat)
	
	lsdat = dat[counter:len(dat)]
	split_data.append(lsdat)

	return split_data


# ------------------------------------------------------------------------------------------------------






def main():
	infile = open('resultfile.txt', 'w')  
	for j in xrange ( 0, 1):
		 
		infile.write('\n\n\npr_num : ' + str(j) + '---------------------------------------\n')        
    
		problem =  j# [1,2,3] choose your problem (sunspot data)
		
		if problem == 0:
	 	   TrDat  = np.loadtxt("sun_train.csv", delimiter=',') # sunspot
		   TesDat  = np.loadtxt("sun_test.csv", delimiter=',') #  
	  	   Hidden = 3
		   Input = 4
		   Output = 1
		   learnRate = 0.1 
		   mRate = 0.01   
		   TrainData  = normalisedata(TrDat, Input, Output) 
		   TestData  = normalisedata(TesDat, Input, Output)  
		   MaxTime = 200
		
		   
	     

	    

		base = [Input, Hidden, Output] # define base network topology. multi-task learning will build from this base module which will have its knowledge shared across the rest of the modules. 
	       
		
		# in this example, we only increase hidden neurons for different tasks given by respective modular topologies

		MaxRun = 1 # number of experimental runs 
		 
		MinCriteria = 95 #stop when learn error less than 0.0001
		
		  
		
		numModules = 10 # first decide number of  modules (or ensembles for comparison)
		
		mtaskNet =   np.array([base, base,base,base,base, base,base,base,base, base,base, base,base,base,base, base,base,base,base, base])
		
		trainPerf = np.random.randn(MaxRun,numModules)  
		testPerf =  np.random.randn(MaxRun,numModules) 
		meanTrain =  np.zeros(numModules)
		stdTrain =  np.zeros(numModules)         
		meanTest =  np.zeros(numModules)
		stdTest =  np.zeros(numModules)

		trainMSE =  np.random.randn(MaxRun) 
		testMSE =  np.random.randn(MaxRun) 
		Epochs =  np.zeros(MaxRun)
		Time =  np.zeros(MaxRun)
		 
		
		
		transKnow = 0
	
		for run in xrange(0, MaxRun  ):  
			mt = MTnetwork(mtaskNet, TrainData, TestData, MaxTime,MinCriteria,learnRate, numModules,transKnow)  
			( trainMSE[run], testMSE[run],trainPerf[run],testPerf[run]) = mt.Procedure()
			#print testPerf[run,:]
	 

		#for module in xrange(0, numModules ):      
    		meanTrain = np.mean(trainMSE) 
    		stdTrain= np.std(trainMSE)
    		meanTest = np.mean(testMSE) 
    		stdTest = np.std(testMSE)
    		
    		meanTrainPerf = np.mean(trainPerf) 
    		stdTrainPerf = np.std(trainPerf)
    		meanTestPerf = np.mean(testPerf) 
    		stdTestPerf = np.std(testPerf)

		#print meanTrainPerf , '+-' , stdTrainPerf
		#print meanTestPerf, '+-', stdTestPerf
		print trainMSE
		print testMSE
		print meanTrain , '+-' , stdTrain
		print meanTest, '+-', stdTest
		
		tr_result = str(meanTrain) + ' +- ' + str(stdTrain)
		infile.write(tr_result+ '\n' )
		ts_result = str(meanTest) + ' +- '+ str(stdTest)
		infile.write(ts_result)
		
		#tr_result = str(meanTrainPerf) + ' +- ' + str(stdTrainPerf)
		#infile.write(tr_result+ '\n' )
		#ts_result = str(meanTestPerf) + ' +- '+ str(stdTestPerf)
		#infile.write(ts_result)
	  
         
 	#plt.figure()
	#plt.plot(erPlot) # plots all the net modules from the last exp run
	#plt.ylabel('error')  
        #plt.savefig('out.png')
       
 
if __name__ == "__main__": main()


