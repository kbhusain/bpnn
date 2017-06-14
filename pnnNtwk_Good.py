#
# Back propagation neural network (BPNN)
# 
# 
import numarray
import math
import random
import string

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):  return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
	return numarray.resize(numarray.ones(I * J,'f') * fill,(I,J))

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
	return math.tanh(x)

# derivative of our sigmoid function
def dsigmoid(y):
	return 1.0-y*y
#
# The class itself. 
# 
class myBPNN:
	def __init__(self, ni, nh, no):
		"""
		Inputs: number of input, hidden, and output nodes
		"""
		self.ni = ni + 1 # +1 for bias node
		self.nh = nh
		self.no = no

		# activations for nodes
		self.ai = numarray.ones(self.ni,'f')
		self.ah =  numarray.ones(self.nh,'f')
		self.ao =  numarray.ones(self.no,'f')
		
		# create weights
		self.wi = makeMatrix(self.ni, self.nh)
		self.wo = makeMatrix(self.nh, self.no)
		# set them to random vaules
		for i in range(self.ni):
			for j in range(self.nh):
				self.wi[i][j] = rand(-2.0, 2.0)
				
		for j in range(self.nh):
			for k in range(self.no):
				self.wo[j][k] = rand(-2.0, 2.0)

		# last change in weights for momentum   
		self.ci = makeMatrix(self.ni, self.nh)
		self.co = makeMatrix(self.nh, self.no)

	def update(self, inputs):
		if len(inputs) != self.ni-1: raise ValueError, 'wrong number of inputs %d %d ' % (len(inputs), self.ni)
		self.ai[:-1] = inputs
		self.ah = map(sigmoid, numarray.matrixmultiply(self.ai, self.wi))
		self.ao = map(sigmoid, numarray.matrixmultiply(self.ah, self.wo))
		return self.ao[:]

	def backPropagate(self, targets, N, M):
		# N is the effect of the current situation = function of adaptability
		# M is the momentum gained by learning = function of conservation of learning
		if len(targets) != self.no:	raise ValueError, 'wrong number of target values'
		hidden_deltas = numarray.zeros(self.nh,'f') * 0.0; #[0.0] * self.nh
		output_deltas = numarray.zeros(self.no,'f') * 0.0; #  [0.0] * self.no
		# ------------------- Errors per output vector.
		for k in range(self.no):
			# diff_a = targets - self.ao
			output_deltas[k] = dsigmoid(self.ao[k]) * (targets[k]-self.ao[k])
		#output_deltas = map(lambda v,d: dsigmoid(v)*d, self.ao, targets - self.ao)	

		# ------------------- Errors - delta per output vector. 
		for j in xrange(self.nh):
			error = 0.0
			# vec = numarray.transpose(output_deltas[k,:]) * self.wo[j,:]
			# error += reduce(vec)
			for k in range(self.no):
				error = error + output_deltas[k]*self.wo[j][k]
			hidden_deltas[j] = dsigmoid(self.ah[j]) * error
		# update output weights
		for j in range(self.nh):
			for k in range(self.no):
				change = output_deltas[k]*self.ah[j]
				self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
				self.co[j][k] = change
				#print N*change, M*self.co[j][k]
		# update input weights
		for i in range(self.ni):
			for j in range(self.nh):
				change = hidden_deltas[j]*self.ai[i]
				self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
				self.ci[i][j] = change          # keep change. 
		# calculate total error
		error = 0.0
		for k in range(len(targets)):
			error += 0.5*(targets[k]-self.ao[k])**2
		return error


	def teste(self, patterns, debug=0):
		"""
		"""
		k    = 0
		ksol = 0
		vsol = []
		kerr = 100000.00              # Absurd number
		for p in patterns:            # For all patterns,
			v = self.update(p[0])     # Try to get an update.
			err = abs(v[0] - p[1][0])  # How close are you? 
			print "ERR", err, v, p[1][0]
			if err < kerr:
				kerr = err          # This is one solution
				vsol = v[:]       # 
				ksol = k
				print "-->CORRECTED ERR", p[0], '->', v, kerr, patterns[ksol], ksol
			k = k + 1                 # Track
		print "finale " , ksol, vsol, 
		return ksol, vsol[0]
	# Debug
	def weights(self):
		print 'Input weights:'
		for i in range(self.ni): print self.wi[i]
		print
		print 'Output weights:'
		for j in range(self.nh): print self.wo[j]

	def train(self, patterns, iterations=1000, N=0.5, M=0.1):
		# N: learning rate
		# M: momentum factor
		for i in xrange(iterations):
			error = 0.0
			for p in patterns:
				inputs, targets, names = p
				self.update(inputs)
				error = error + self.backPropagate(targets, N, M)
			if i % 100 == 0: print 'Training error %-14f' % error, i, " of ", iterations  

	def test(self, patterns, ctr):
		for p in patterns:            # For all patterns,
			for x in range(ctr): 
				v = self.update(p[0])     # Try to get an update.
				err = abs(v[0] - p[1][0])  # How close are you? 
			err = err / ctr
			print "ERR", err, v, p[1][0], p[2][0]
		

def demo():
	# Teach network XOR function
	pat = [
		[[0,0], [0], ['a']],
		[[0,1], [1], ['b']],
		[[1,0], [1], ['c']],
		[[1,1], [0], ['d']]
	]

	n = myBPNN(2, 2, 1)
	n.train(pat)
	n.test(pat,10)

def demo2():
	pat = [ 
		[[0.33333000000000002, 0.33333000000000002], [ 0.657320912741, ], [ 40.154389 ]],
		[[0.33333000000000002, 0.49831151515151517], [ 0.666666666667, ], [ 41.207688 ]],
		[[0.33333000000000002, 0.66666000000000003], [ 0.661931702003, ], [ 40.674041 ]],
		[[0.33934196392785571, 0.33333000000000002], [ 0.415515116166, ], [ 12.902034 ]],
		[[0.33934196392785571, 0.36363272727272727], [ 0.455410432265, ], [ 17.398375 ]],
		[[0.33934196392785571, 0.49831151515151517], [ 0.410720561506, ], [ 12.361671 ]],
		[[0.33934196392785571, 0.66666000000000003], [ 0.426095064446, ], [ 14.094431 ]],
		[[0.39946160320641283, 0.33333000000000002], [ 0.419236403177, ], [ 13.321436 ]],
		[[0.39946160320641283, 0.36363272727272727], [ 0.378465444204, ], [ 8.726407 ]],
		[[0.39946160320641283, 0.49831151515151517], [ 0.36969362996, ], [ 7.737793 ]],
		[[0.39946160320641283, 0.66666000000000003], [ 0.412593981902, ], [ 12.572812 ]],
		[[0.66666000000000003, 0.33333000000000002],  [ 0.333333333333, ], [ 3.639861 ]],
		[[0.66666000000000003, 0.36363272727272727], [ 0.478093023586, ], [ 19.954782 ]],
		[[0.66666000000000003, 0.49831151515151517], [ 0.523127444484, ], [ 25.030318 ]],
		[[0.66666000000000003, 0.66666000000000003], [ 0.542653230028, ], [ 27.230942 ]]
	]

	test_pat = [
		[[0.01, 0.33], [ 0.333, ], [ 11 ]], 
		[[0.11, 0.33], [ 0.333, ], [ 12 ]], 
		[[0.21, 0.33], [ 0.333, ], [ 13 ]], 
		[[0.31, 0.33], [ 0.333, ], [ 14 ]], 
		[[0.41, 0.33], [ 0.333, ], [ 14 ]], 
		[[0.52, 0.33], [ 0.333, ], [ 15 ]], 
		[[0.63, 0.33], [ 0.333, ], [ 16 ]], 
		[[0.64, 0.33], [ 0.333, ], [ 17 ]], 
		[[0.65, 0.33], [ 0.333, ], [ 18 ]], 
		[[0.66, 0.33], [ 0.333, ], [ 19 ]], 
	]
	n = myBPNN(2, 6, 1)
	n.train(pat[11:12])
	n.test(pat,3)
	print "------"
	n.test(test_pat,1)


def demo3():
	pat = [ 
		[[0.33333000000000002, 0.33333000000000002], [ 0.657320912741, ], [ 40.154389 ]],
		[[0.33333000000000002, 0.49831151515151517], [ 0.666666666667, ], [ 41.207688 ]],
		[[0.33333000000000002, 0.66666000000000003], [ 0.661931702003, ], [ 40.674041 ]],
		[[0.33934196392785571, 0.33333000000000002], [ 0.415515116166, ], [ 12.902034 ]],
		[[0.33934196392785571, 0.36363272727272727], [ 0.455410432265, ], [ 17.398375 ]],
		[[0.33934196392785571, 0.49831151515151517], [ 0.410720561506, ], [ 12.361671 ]],
		[[0.33934196392785571, 0.66666000000000003], [ 0.426095064446, ], [ 14.094431 ]],
		[[0.39946160320641283, 0.33333000000000002], [ 0.419236403177, ], [ 13.321436 ]],
		[[0.39946160320641283, 0.36363272727272727], [ 0.378465444204, ], [ 8.726407 ]],
		[[0.39946160320641283, 0.49831151515151517], [ 0.36969362996, ], [ 7.737793 ]],
		[[0.39946160320641283, 0.66666000000000003], [ 0.412593981902, ], [ 12.572812 ]],
		[[0.66666000000000003, 0.33333000000000002],  [ 0.333333333333, ], [ 3.639861 ]],
		[[0.66666000000000003, 0.36363272727272727], [ 0.478093023586, ], [ 19.954782 ]],
		[[0.66666000000000003, 0.49831151515151517], [ 0.523127444484, ], [ 25.030318 ]],
		[[0.66666000000000003, 0.66666000000000003], [ 0.542653230028, ], [ 27.230942 ]]
	]

	pat2 = []
	for a in pat: 
		pat2.append([a[1],a[0],a[2]])
		print pat2

	n = myBPNN(1, 6, 2)
	n.train(pat2)
	n.test(pat2,3)

if __name__ == '__main__':
	demo2()
