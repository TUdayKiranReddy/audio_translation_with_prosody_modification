import numpy as np

def dtw_allign(key_t,sig_t):

 
  key = key_t
  sig = sig_t
  N = len(key)
  M = len(sig)

  ldm = np.power(key.reshape(1, -1) - sig.reshape(-1, 1), 2)

  adm = np.zeros((M,N))
  adm[:,0] = ldm[:,0]
  for n in range(1, N):
    col = adm[:,n-1]
    most = np.max(col) + 1
    col = np.hstack((most, most, col)).reshape(-1, 1)
    new = np.hstack((col[0:M], col[1:M+1], col[2:M+2]))
    minimums = np.min(new, axis=1)
    adm[:,n] = ldm[:,n] + minimums

  w = np.zeros(N)
  w[N-1] = M
  w[0] = 1
  r = M
  flatflag = 0
  jumpflag = 0
  for c in range(N-2, 0,-1):
    if (r >= 3) and (not jumpflag):
      choices = [adm[r-1,c], adm[r-2,c], adm[r-3,c]]
    elif (r >= 3) and jumpflag:
      choices = [adm[r-1,c], adm[r-2,c]]
      jumpflag = 0
    elif (r == 2):
      choices = [adm[r-1,c], adm[r-2,c]]
    elif (r == 1):
      choices = [adm[r-1,c]]

    if flatflag & (r != 1):
      choices = choices[1:len(choices)]
      posn = np.max(np.where(choices==np.min(choices)))+1
      if posn == len(choices):  
        jumpflag == 1
      flatflag = 0
    else:
      posn = np.max(np.where((choices==np.min(choices))))
      if posn == 2:
        jumpflag = 1
      if posn == 0:
        flatflag = 1

    r = r - posn
    w[c] = r
  path = (w-1).astype(np.int)
  return path