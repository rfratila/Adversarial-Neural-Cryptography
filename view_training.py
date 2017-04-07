import numpy
import pylab #for graphing
import pickle
import sys

if len(sys.argv) > 1:
    modelName = sys.argv[1]
with open("{}".format(modelName), 'rb') as infile:
    if '.pickle' in modelName:
        data = pickle.load(infile)

pylab.plot(data['iteration'],data['eve_error'], '-ro',label='Eve Error')
pylab.plot(data['iteration'],data['bob_error'],'-go',label='Bob Error')
pylab.xlabel("iteration")
pylab.ylabel("Error")
#pylab.ylim(0,max(data['eve_error'])
pylab.title(modelName)
pylab.legend(loc='upper right')
#pylab.savefig('.png'%modelName)
pylab.show()#enter param False if running in iterative mode