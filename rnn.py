import math
import numpy as np
import matplotlib.pyplot as plt

def sig(x):
    return 1 / (1 + math.exp(-x))

def sigPrime(x):
    return sig(x) * (1 - sig(x))

def error(y,t):
    return .5 * (y - t) ** 2

def errorPrime(y, t):
    return y - t

def out(a):
    return a

def outPrime(a):
    return 1

class RNN:
    
    def __init__(self, No = 1, Nh = 20, Ni = 5, weightSig = .01, timeDepth = 4, lr = .0005):
        self.No = No
        self.Nh = Nh
        self.Ni = Ni    
        self.weightSig = weightSig
        self.timeDepth = timeDepth
        self.lr = lr

        self.netFunc = np.vectorize(sig)
        self.outFunc = np.vectorize(out)
        self.netFuncPrime = np.vectorize(sigPrime)
        self.outFuncPrime = np.vectorize(outPrime)
        self.errorFuncPrime = np.vectorize(errorPrime)
        self.errorFunc = np.vectorize(error)

        self.initWeights()
        self.updateInput()

    def initWeights(self):
        self.bh = np.zeros((self.Nh,1))
        self.bo = np.zeros((self.No,1))
        self.wnet = np.random.normal(0,self.weightSig,(self.Nh, self.Nh))
        self.wout = np.random.normal(0,self.weightSig,(self.No, self.Nh))
        self.win = np.random.normal(0,self.weightSig,(self.Nh, self.Ni))
        self.h0 = np.zeros((self.Nh,1))

    def updateInput(self):
        self.inp = np.zeros((self.timeDepth, self.Ni,1))
        self.d = np.zeros((self.timeDepth, self.No, 1))
        n = np.random.randint(1, self.Ni+1)
        self.inp[1][n-1] = 1
        self.d[1] = n
        for i in range(2, self.timeDepth):
            n = np.random.randint(1, self.Ni+1)
            self.inp[i][n-1] = 1
            self.d[i] = self.d[i-1] * n
                
        #self.d = np.zeros((self.timeDepth, 1, 1))
        #self.d[0] = self.inp[0]
        #for i in range(1,len(self.d)):
        #    self.d[i] = self.d[i-1] * self.inp[i]
        #print(self.d)

    def resetGrad(self):
        self.do = np.zeros((self.timeDepth, self.No,1))
        self.dh = np.zeros((self.timeDepth, self.Nh,1))
        self.du = np.zeros((self.timeDepth, self.Nh,1))
        self.dwnet = np.zeros((self.Nh, self.Nh))
        self.dwout = np.zeros((self.No, self.Nh))
        self.dwin = np.zeros((self.Nh, self.Ni))
        self.dbh = np.zeros((self.Nh,1))
        self.dbo = np.zeros((self.No,1))



    def resetState(self):
        self.z = np.zeros((self.timeDepth, self.No,1))
        self.o = np.zeros((self.timeDepth, self.No,1))
        self.h = np.zeros((self.timeDepth, self.Nh, 1))
        self.h[0] = self.h0
        self.u = np.zeros((self.timeDepth, self.Nh,1))


    def forward(self):
        self.resetState()
        for t in range(1, self.timeDepth):
            self.u[t] = np.dot(self.win, self.inp[t]) + np.dot(self.wnet, self.h[t-1]) + self.bh
            self.h[t] = self.netFunc(self.u[t])
            self.o[t] = np.dot(self.wout, self.h[t]) + self.bo
            self.z[t] = self.outFunc(self.o[t])
    
    def uCalc(self):    
        for t in range(1, self.timeDepth):
            self.u[t] = np.dot(self.win, self.inp[t]) + np.dot(self.wnet, self.h[t-1]) + self.bh
            self.h[t] = self.netFunc(self.u[t])
            self.o[t] = np.dot(self.wout, self.h[t]) + self.bo
            self.z[t] = self.outFunc(self.o[t])
    def hCalc(self):    
        for t in range(1, self.timeDepth):
            self.h[t] = self.netFunc(self.u[t])
            self.o[t] = np.dot(self.wout, self.h[t]) + self.bo
            self.z[t] = self.outFunc(self.o[t])
    def oCalc(self):    
        for t in range(1, self.timeDepth):
            self.o[t] = np.dot(self.wout, self.h[t]) + self.bo
            self.z[t] = self.outFunc(self.o[t])
    def zCalc(self):    
        for t in range(1, self.timeDepth):
            self.z[t] = self.outFunc(self.o[t])
    def errorPrimeCalc(self):
        self.EP = 0.0
        for t in range(1, self.timeDepth):
            self.EP += self.errorFuncPrime(self.z[t],self.d[t])
        return self.EP
    
    def errorCalc(self):
        self.E = 0.0
        for t in range(1, self.timeDepth):
            self.E += self.errorFunc(self.z[t],self.d[t])
        return self.E
        
    def backward(self):
        self.resetGrad()
        for t in reversed(range(1, self.timeDepth)):
            self.do[t] = self.errorFuncPrime(self.z[t], self.d[t])#self.outFuncPrime(self.o[t]) *
            self.dbo += self.do[t]
            self.dwout += np.dot(self.do[t], self.h[t].transpose())
            self.dh[t] += np.dot(self.wout.transpose(), self.do[t])
            self.du[t] = self.netFuncPrime(self.u[t]) * self.dh[t]
            self.dwin += np.dot(self.du[t], self.inp[t].transpose())
            self.dbh += self.du[t]
            self.dwnet += np.dot(self.du[t], self.h[t-1].transpose())
            self.dh[t-1] = np.dot(self.wnet.transpose(), self.du[t])

    def stepBackward(self,t):
        print(self.d)
        print(self.z[t])
        if t in self.trainDepth:
            self.do[t] = self.outFuncPrime(self.errorFuncPrime(self.z[t], self.d))
            print(self.do[t])
        else:
            self.do[t] = np.zeros((self.No, 1))
            print(self.do[t])
        self.dbo[:] = self.dbo[:] + self.do[t]
        print(self.dbo[:])
        self.dwout[:] = self.dwout[:] + self.do[t] * self.h[t]
        print(self.dwout[:])
        self.dh[t] = self.dh[t] + np.dot(self.wout.transpose(), self.do[t])
        print(self.dh[t])
        self.du[t] = self.netFuncPrime(self.u[t]) * self.dh[t].wout
        print(self.du[t])
        self.dwin[:] = self.dwin[:] + np.dot(self.du[t], self.inp[t].transpose())
        print(self.dwin[:])
        self.dbh[:] = self.dbh[:] + self.du[t]
        print(self.dbh[:])
        self.dwnet[:] = self.dwnet[:] + np.dot(self.du[t], self.h[t-1].transpose())
        print(self.dwnet[:])
        self.dh[t-1] = np.dot(self.wnet.transpose(), self.du[t])
        print(self.dh[t-1])

    def weightTest(self, delta):
        #self.initWeights()
        #self.updateInput()
        self.forward()
        self.errorCalc()
        f = self.E
        print(f)
        self.backward()
        d = self.dwin[0]
        self.win[0] += delta
        self.uCalc()
        self.errorCalc()
        fp = self.E
        print(fp)
        dr = (fp - f) / delta
        self.win[0] -= delta
        return d, dr
            
    def update(self):
        for dparam in [self.dwnet, self.dwout, self.dwin, self.dbo, self.dbh]:
            np.clip(dparam, -5, 5, out=dparam)
        self.wnet -= self.lr * self.dwnet
        self.wout -=  self.lr * self.dwout
        self.win -=  self.lr * self.dwin
        self.bo -=  self.lr * self.dbo
        self.bh -=  self.lr * self.dbh
        self.h0 -= self.lr * self.dh[0]

    def run(self):
        vals=[]
        for i in range(5000000):
            self.updateInput()
            self.forward()
            self.errorCalc()
            self.backward()
            self.update()
            if i%10000 == 0:
                vals.append(self.errorFunc(self.z[self.timeDepth-1],self.d[self.timeDepth-1])[0][0])
                print('in: ')
                print(self.d)
                print('out:')
                print(self.z)
                print('error:')
                print(self.E)
                print()

        plt.plot(vals)
        plt.show()

#forward()
#backward()
#update()
'''
def forward(input, timeLength):
    output[n][0] = sig(np.dot(input[t], win))
    for t in range(1,timeLength):
        for n in networkSize:
            nodeInput[n][t] = np.dot(input[t], win) + np.dot(output[:][t-1], wnet)
            output[n][t] = sig(nodeInput[n][t])


def backward(target, timeLength):
    for n in networkSize:
        gwout[n][timeLength-1] = errorPrime(output[n][timeLength-1], target[n][timeLength-1]) * sigPrime(nodeInput[n][timeLength-1])
    for n in networkSize:
            errorSum = 0.0
            for nn in networkSize:
                errorSum += wout[n][nn] * gwout[nn][timeLength-1]
            gwnet[n][timeLength-1] = errorSum * sigPrime(nodeInput[n][timeLength-1])
    for t in range(timeLength-2, 0, -1):
        for n in networkSize:
            errorSum = 0.0
            for nn in networkSize:
                errorSum += wback[n][nn] * gwnet[nn][t+1]
            gwout[n][t-1] = (errorPrime(output[n][t-1], target[n][t-1]) + errorSum) * sigPrime(nodeInput[n][t])
        for n in networkSize:
        
        errorSum = 0.0
            errorSum2 = 0.0
            for nn in networkSize:
                errorSum += wout[n][nn] * gwout[nn][t]
                errorSum2 += wnet[n][nn] * gwout[nn][t+1]
            gwnet[n][t] = (errorSum + errorSum2) * sigPrime(nodeInput[n][t])
    
'''
            
if __name__ == "__main__":
    a = RNN()
    print(a.weightTest(.0001))
'''a.updateInput()
    a.forward()
    a.stepBackward(1)
    print()
    print()
    a.stepBackward(0)'''

