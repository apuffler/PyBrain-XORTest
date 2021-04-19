__author__ = 'rin'

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
#2x3x1
#net = buildNetwork(2,3,1)
#2 dim in, 1 dim
ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))


ds2 = SupervisedDataSet(2, 1)
ds2.addSample((0, 0), (0,))
ds2.addSample((0, 1), (1,))
ds2.addSample((1, 0), (1,))
ds2.addSample((1, 1), (1,))



#for inpt, target in ds:
#    print inpt, target


#print(net.activate([2, 1]))
net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)
net2 = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)






trainer = BackpropTrainer(net, ds)
trainer2 = BackpropTrainer(net, ds2)
##trainer.train()

for i in range(0,300):
    trainer.train()



print("=======================")
x = net.activate([1,1])
print("should be 0:" + str(round(x,1)))

x = net.activate([0,1])
print("should be 1:" + str(round(x,1)))

x = net.activate([1,0])
print("should be 1:" + str(round(x,1)))

x = net.activate([0,0])
print("should be 0:" + str(round(x,1)))
#for i in range(0,50):
#    trainer.trainUntilConvergence()
#    trainer2.trainUntilConvergence()



#print(net2.activate([0,0]))
#print(net.activate([2, 1])  net.activate([2, 1]))



