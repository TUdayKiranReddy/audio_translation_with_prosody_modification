import numpy as np
import matplotlib.pyplot as plt

def glotlf(d, t=[], p=[]):
#GLOTLF   Liljencrants-Fant glottal model U=(D,T,P)
# d is derivative of flow waveform: must be 0, 1 or 2
# t is a vector of time instants at which to calculate the
#   waveform. Time units are in fractions of a cycle.
# p is a vector of 3 parameters defining the waveform
#    p[0] is the time at which ugd has its peak negative value. This we define as the
#         start of the closed phase. p(1) is therefore the open/closed interval ratio.
#    p[1] is the reciprocal of the peak negative value of ugd(t)
#    p[2] is the fraction of the open phase for which ugd(t) is negative. That is, it is
#         it is the time between the peak flow and the end of the open phase expressed
#	      as a fraction of the open phase.
# Note: this signal has not been low-pass filtered
# and will therefore be aliased [this is a bug]

	if t==[]:
		tt = np.arange(100)/100
	else:
		tt = t - np.floor(t)
	u = np.zeros(tt.shape)
	de = np.array([0.6, 0.1, 0.2])
	if p==[]:
		p=de
	elif len(p) < 2:
		p = np.hstack((p, de[len(p), 2]))
		print(p)

	te = p[0]
	mtc = te-1
	e0=1
	wa = np.pi/(te*(1-p[2]))
	a = -1*np.log(-1*p[1]*np.sin(wa*te))/te
	inta=e0*((wa/np.tan(wa*te)-a)/p[1]+wa)/(np.power(a, 2)+np.power(wa, 2))

	rb0=p[1]*inta
	rb=rb0

	for _ in range(1, 5):
	  kk=1-np.exp(mtc/rb)
	  err=rb+mtc*(1/kk-1)-rb0
	  derr=1-(1-kk)*np.power((mtc/rb/kk), 2)
	  rb=rb-err/derr

	e1=1/(p[1]*(1-np.exp(mtc/rb)))


	ta=tt<te
	tb=np.logical_not(ta)
	if d==0:
	  u[ta]=e0*(np.exp(a*tt[ta])*(a*np.sin(wa*tt[ta])-wa*np.cos(wa*tt[ta]))+wa)/(np.power(a, 2)+np.power(wa, 2))
	  u[tb]=e1*(np.exp(mtc/rb)*(tt[tb]-1-rb)+np.exp((te-tt[tb])/rb)*rb)
	elif d==1:
	  u[ta]=e0*np.exp(a*tt[ta])*np.exp(wa*tt[ta])
	  u[tb]=e1*(np.exp(mtc/rb)-np.exp((te-tt[tb])/rb))
	elif d==2:
	  u[ta]=e0*np.exp(a*tt[ta])*(a*np.sin(wa*tt[ta])+wa*np.cos(wa*tt[ta]))
	  u[tb]=e1*np.exp((te-tt[tb])/rb)/rb
	else:
	  print('Derivative must be 0,1 or 2')
	return u

ncyc=5
period=80
t=np.arange(0, ncyc+1/period, 1/period)
ug=glotlf(0,t)

plt.figure()
plt.plot(t,ug)
plt.xlabel('t')
plt.ylabel('Glottal Flow')
plt.show()