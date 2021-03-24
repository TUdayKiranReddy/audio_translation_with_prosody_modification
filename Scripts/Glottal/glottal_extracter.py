from scipy.io.wavfile import read
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pysptk
from peakdetect import peakdetect
from scipy.integrate import cumtrapz
from scipy.stats import pearsonr
from scipy.linalg import toeplitz
try:
    from scipy.signal.windows import medfilt, hann, filtfilt, blackman, hamming, buttord, butter, lfiltic, lfilter
except:
    from scipy.signal import medfilt, hann, filtfilt, blackman, hamming, buttord, butter, lfiltic, lfilter

def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def create_continuous_smooth_f0(F0,VUV,x):
    #Function to create and f0 contour which is interpolated over unvoiced
    # regions and is heavily smoothed with median and moving average filters.
    # Initial settings
    F0=np.asarray(F0)
    med_len=17
    sm_len=17
    F02=F0[:]
    f0nzp=np.where(F0>0)[0]
    f0nz=F0[f0nzp]
    if len(f0nz)<med_len:
        F0min=np.min(f0nz)
        F0max=np.max(f0nz)
    else:
        F0min=np.min(medfilt(f0nz,med_len))
        F0max=np.max(medfilt(f0nz,med_len))

    posmin=np.where(F0<F0min)[0]
    posmax=np.where(F0>F0max)[0]
    F02[posmin]=F0min
    F02[posmax]=F0max

    f0_inter=F02[:]

    N=len(F0)

    start=[]
    stop=[]
    initV=np.where(VUV==1)[0]

    for j in range(initV[0],N-1):
        if VUV[j]==0 and VUV[j+1]==1:
            stop.append(j)
        elif VUV[j]==1 and VUV[j+1]==0:
            start.append(j)
    n_seg=np.min([len(start),len(stop)])
    for j in range(n_seg):
        f0_int_cur=np.interp(np.arange(start[j], stop[j]), [start[j], stop[j]], [F0[start[j]],F0[stop[j]+1]])
        f0_inter[start[j]+1:stop[j]+1]=f0_int_cur
    f0_inter=smooth(medfilt(f0_inter,med_len),sm_len)
    f0_samp=np.interp(np.arange(len(x)),np.linspace(0,len(x),len(f0_inter)),f0_inter)

    return f0_inter, f0_samp



def GetLPCresidual(wave,L,shift,order=24,VUV=1):

    """
    Get the LPC residual signal
    Written originally by Thomas Drugman, TCTS Lab.

    Adapated to python by
    J. C. Vasquez-Correa
    Pattern recognition Lab, University of Erlangen-Nuremberg
    Faculty of Enginerring, University of Antiqouia,

    :param wave: array with the speech signal
    :param L: window length (samples) (typ.25ms)
    :param shift: window shift (samples) (typ.5ms)
    :param order: LPC order
    :param VUV: vector of voicing decisions (=0 if Unvoiced, =1 if Voiced)
    :returns res: LPC residual signal 
    """
    start=0
    stop=int(start+L)
    res=np.zeros(len(wave))
    n=0
    while stop<len(wave):

        if np.sum(VUV[start:stop])==len(VUV[start:stop]): # if it is avoiced segment
            segment=wave[start:stop]
            segment=segment*hann(len(segment))
            try:
                A=pysptk.sptk.lpc(segment, order)
                inv=filtfilt(A,1,segment)
                inv=inv*np.sqrt(np.sum(segment**2)/np.sum(inv**2))
                res[start:stop]=inv
            except:
                print("WARNING: LPCs cannot be extracted for the segment")
        start=int(start+shift)
        stop=int(stop+shift)
        n=n+1
    res=res/max(abs(res))
    return res



def RCVD_reson_GCI(res,fs,F0mean):

    # Function to use the resonator used in RCVD (creaky voice detection),
    # applied to the LP-residual signal and give output

    ## Configure resonator (using settings in RCVD)

    Phi=2*np.pi*1*F0mean/fs
    Rho=0.9 # Set to a narrow bandwidth
    rep=filtfilt([1., 0., 0.],[1., -2*Rho*np.cos(Phi), Rho**2],res, padtype = 'odd', padlen=9) # Filter forwards and backwards to have zero-phase
    y=rep/np.max(np.abs(rep))
    return y


def zeroPhaseHPFilt(x,fs,f_p,f_s):
    Rp=0.5
    Rs=40
    Wp=f_p/(fs/2.)
    Ws=f_s/(fs/2.)
    [n,Wn] = buttord(Wp,Ws,Rp,Rs)
    [b,a]=butter(n,Wn,'high')
    y = filtfilt(b,a,x, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    return y


def get_MBS(x,fs,T0mean):
    # Obtain the mean-based signal
    MBS=np.zeros(len(x))
    halfL=int(1.6*T0mean[0]/2)

    StepExp=3
    Step=2**StepExp
    for m in range(halfL, len(x)-halfL, Step):
        if len(T0mean)==1:
            halfL=int(1.7*T0mean[0]/2)
        else:
            halfL=int(1.7*T0mean[m]/2)
        Blackwin=blackman(2*halfL)
        start=int(m-halfL)
        stop=int(m+halfL)
        if stop>len(x):
            break
        if start>0:
            vec=x[start:stop]*Blackwin
            MBS[m]=np.mean(vec)

    t=np.where(MBS!=0)[0]
    MBS=np.interp(np.arange(len(x)), t, MBS[t])
    MBS[np.isnan(MBS)]=0
    MBS=zeroPhaseHPFilt(MBS,fs, 70,10)
    MBS=MBS/max(abs(MBS))
    MBS=smooth(MBS,7)
    return MBS


def get_MBS_GCI_intervals(MBS,fs,T0mean,F0max=500):
    F0max=F0max*2
    T0max=int(fs/F0max)
    [max_peaks, min_peaks]=peakdetect(MBS,lookahead = T0max)

    idx=np.asarray([min_peaks[j][0] for j in range(len(min_peaks))])
    N=len(idx)
    search_rate=0.28
    search_left_rate=0.01
    interval=np.zeros((N,2))

    for n in range(N):
        if len(T0mean)>1:
            start=idx[n]-int(T0mean[idx[n]]*search_left_rate)
            stop=idx[n]+int(T0mean[idx[n]]*search_rate)
        else:
            start=idx[n]-int(T0mean*search_left_rate)
            stop=idx[n]+int(T0mean*search_rate)
        if start<1:
            start=1
        if stop>len(MBS) and start<len(MBS):
            stop=len(MBS)
        elif stop>len(MBS) and start>=len(MBS):
            break
        interval[n,0]=start
        interval[n,1]=stop
    return interval


def search_res_interval_peaks(res,interval,Ncand,VUV):
    #Function to search for Ncand peaks in the residual within each search
    #interval

    N=len(interval)
    GCI=np.zeros((N,Ncand))
    rel_amp=np.zeros((N,Ncand))
    GCI_cur=0
    for n in range(N):
        start=int(interval[n][0])
        stop=int(interval[n][1])
        if stop<=start or np.sum(VUV[start:stop])!=len(VUV[start:stop]):
            continue
        if stop-start<Ncand:
            amp=np.max(res[start:stop])
            idx=np.argmax(res[start:stop])
            GCI_cur=GCI_cur+start-1
            GCI[n,:]=GCI_cur
            rel_amp[n,:]=0
        else:
            amp=np.sort(res[start:stop])
            amp=amp[::-1]
            idx=np.argsort(res[start:stop])
            idx=idx[::-1]
            GCI_cur=idx[0:Ncand]+start-1
            GCI[n,:]=np.asarray(GCI_cur)

            if max(amp[0:Ncand])>0:
                rel_amp[n,:]=1-(amp[0:Ncand]/max(amp[0:Ncand]))
            else:
                rel_amp[n,:]=np.ones(Ncand)

    GCI=[GCI[j] for j in range(len(GCI)) if sum(GCI[j])>0]
    rel_amp=[rel_amp[j] for j in range(len(GCI)) if sum(GCI[j])>0]
    return GCI, rel_amp


def RESON_dyProg_mat(GCI_relAmp,GCI_N,F0mean,x,fs,trans_wgt,relAmp_wgt, plots=True):
    # Function to carry out dynamic programming method described in Ney (1989)
    # and used previously in the ESPS GCI detection algorithm. The method
    # considers target costs and transition costs which are accumulated in
    # order to select the `cheapest' path, considering previous context

    # USAGE: INPUT
    #        GCI_relAmp - target cost matrix with N rows (GCI candidates) by M
    #                     columns (mean based signal derived intervals).
    #        GCI_N      - matrix containing N by M candidate GCI locations (in
    #                     samples)
    #        F0_inter   - F0 values updated every sample point
    #        x          - speech signal
    #        fs         - sampling frequency
    #        trans_wgt  - transition cost weight
    #
    #        OUTPUT
    #        GCI        - estimated glottal closure instants (in samples)
    # =========================================================================
    # === FUNCTION CODED BY JOHN KANE AT THE PHONETICS LAB TRINITY COLLEGE ====
    # === DUBLIN. 25TH October 2011 ===========================================
    # =========================================================================

    # =========================================================================
    # === FUNCTION ADAPTED AND CODED IN PYTHON BY J. C. Vasquez-Correa
    #   AT THE PATTERN RECOGNITION LAB, UNIVERSITY OF ERLANGEN-NUREMBERG ====
    # === ERLANGEN, MAY, 2018 ===========================================
    # =========================================================================



    ## Initial settings

    GCI_relAmp=np.asarray(GCI_relAmp)
    relAmp_wgt=np.asarray(relAmp_wgt)
    cost = GCI_relAmp*relAmp_wgt
    #print(cost.shape)
    GCI_N=np.asarray(GCI_N)
    ncands=GCI_N.shape[1]
    nframe=GCI_N.shape[0]
    #print(ncands, nframe, cost.shape)
    prev=np.zeros((nframe,ncands))
    pulseLen = int(fs/F0mean)
    GCI_opt=np.zeros(nframe)

    for n in range(nframe):

        if n<1:
            continue

        costm=np.zeros((ncands,ncands))

        for c in range(ncands):
            #Transitions TO states in current frame
            start=int(GCI_N[n,c]-pulseLen/2)
            stop=int(GCI_N[n,c]+pulseLen/2)
            if stop>len(x):
                stop=len(x)
            pulse_cur=x[start:stop]

            for p in range(ncands):
                #Transitions FROM states in previous frame
                start=int(GCI_N[n-1,p]-pulseLen/2)
                stop=int(GCI_N[n-1,p]+pulseLen/2)
                if start<1:
                    start=1
                if stop>len(x):
                    stop=len(x)
                pulse_prev=x[start:stop]
                if len(pulse_cur)==0 or np.isnan(pulse_cur[0]) or np.isnan(pulse_prev[0]):
                    costm[p,c]=0
                else:
                    if len(pulse_cur)!=len(pulse_prev):
                        cor_cur=0
                    else:
                        cor_cur=pearsonr(pulse_cur,pulse_prev)[0]
                    costm[p,c]=(1-np.abs(cor_cur))*trans_wgt

        costm=costm+np.tile(cost[n-1,0:ncands],(ncands,1))
        costm=np.asarray(costm)
        costi=np.min(costm,0)
        previ=np.argmin(costm,0)
        cost[n,0:ncands]=cost[n,0:ncands]+costi
        prev[n,0:ncands]=previ

    best=np.zeros(n+1)
    best[n]=np.argmin(cost[n,0:ncands])
    for i in range(n-1,1,-1):

        best[i-1]=prev[i,int(best[i])]

    for n in range(nframe):
        #print(n,int(best[n]))
        GCI_opt[n]=GCI_N[n,int(best[n])]

    if plots:
        GCI_norm=np.zeros((nframe,ncands))
        GCI_opt_norm=np.zeros((nframe,ncands))
        for n in range(nframe):
            GCI_norm[n,:]=GCI_N[n,:]-GCI_N[n,0]
            GCI_opt_norm[n]=GCI_opt[n]-GCI_N[n,0]

        plt.subplot(211)
        plt.plot(x)
        plt.stem(GCI_N[:,0], -0.1*np.ones(len(GCI_N[:,0])), 'r')
        plt.stem(GCI_opt, -0.1*np.ones(len(GCI_opt)), 'k')
        plt.subplot(212)
        #plt.plot(GCI_opt, GCI_norm)
        plt.plot(GCI_opt, GCI_opt_norm, 'bo')
        plt.show()
    return GCI_opt


def lpcauto(s,p):
    #LPCAUTO  performs autocorrelation LPC analysis [AR,E,K]=(S,P)
    #  Inputs:
    #     s(ns)   is the input signal
    #	   p       is the order (default: 12)
    #
    # Outputs:
    #          ar(nf,p+1) are the AR coefficients with ar(1) = 1
    #          e(nf)      is the energy in the residual.
    #                     sqrt(e) is often called the 'gain' of the filter.

    nf=1
    ar=np.zeros(p+1)
    ar[0]=1
    e=np.zeros(nf)
    dd=s[:]
    nc=len(s)
    pp=min(p,nc)
    ww=np.hamming(nc)
    y=np.zeros(nc+p)
    wd=dd*ww
    y[0:nc]=wd
    z=np.zeros((nc,pp+1))
    for j in range(pp+1):
        z[:,j]=y[j:j+len(s)]
    rr=np.matmul(wd,z)
    rm=toeplitz(rr[0:pp])
    rk=np.linalg.matrix_rank(rm)
    if rk!=0:
        if rk<pp:
            rm=rm[0:rk,0:rk]

        ar[1:rk+1]=np.matmul(-rr[1:rk+1], np.linalg.inv(rm))
    e=np.dot(rr,ar)
    return ar, e


def calc_residual(x,x_lpc,ord_lpc,GCI):

    # Function to carry out LPC analysis and inverse filtering, used in IAIF.m
    # function.

    # USAGE:
    #       Input:
    #             x       : signal to be inverse filtered
    #             x_lpc   : signal to carry out LPC analysis on
    #             ord_lpc : LPC prediction order
    #             GCI     : Glottal closure instants (in samples)
    #
    #       Output:
    #             vector_res : residual after inverse filtering
    #
    #
    #########################################################################
    ## Function Coded by John Kane @ The Phonetics and Speech Lab ###########
    ## Trinity College Dublin, August 2012 ##################################
    #########################################################################

    #########################################################################
    ## Function transalated to Python by J. C. Vasquez-Correa @ University of Erlangen-Nuremberg ###########
    #########################################################################

    vector_res=np.zeros(len(x))
    ze_lpc=np.zeros(ord_lpc)
    ar_lpc=np.zeros((ord_lpc+1, len(GCI)))

    for n in range(len(GCI)-1):
        if n>1:
            T0_cur=GCI[n]-GCI[n-1]
        else:
            T0_cur=GCI[n+1]-GCI[n]

        if GCI[n]-T0_cur>0 and GCI[n]+T0_cur<=len(x) and T0_cur>0:
            start=int(GCI[n]-T0_cur)
            stop=int(GCI[n]+T0_cur)

            frame_lpc=x_lpc[start:stop]
            if len(frame_lpc)>ord_lpc*1.5:
                frame_wind=frame_lpc*hamming(len(frame_lpc))
                ar,e=lpcauto(frame_wind, ord_lpc)
                ar_par=np.real(ar)
                #ar_par[0]=1
                ar_lpc[:,n]=ar_par

            # inverse filtering
            try:
                if n>1 and ('frame_res' in locals()) and ('residual' in locals()):
                    last_input=frame_res[::-1]
                    last_output=residual[::-1]

                    ze_lpc=lfiltic(ar_par, np.sqrt(e), last_output, last_input)
                frame_res=x[start:stop]

                residual=lfilter(b=ar_par,a=np.sqrt(e), x=frame_res, zi=ze_lpc)

            except:
                residual=[frame_res]
            residual_win=residual[0]*hamming(len(residual[0]))
            try:
                vector_res[start:stop]=vector_res[start:stop]+residual_win
            except:
                vector_res[start:start+len(residual_win)]=vector_res[start:start+len(residual_win)]+residual_win

    return vector_res

def SE_VQ_varF0(x,fs, f0=None):
    """
    Function to extract GCIs using an adapted version of the SEDREAMS 
    algorithm which is optimised for non-modal voice qualities (SE-VQ). Ncand maximum
    peaks are selected from the LP-residual signal in the interval defined by
    the mean-based signal. 
    
    A dynamic programming algorithm is then used to select the optimal path of GCI locations. 
    Then a post-processing method, using the output of a resonator applied to the residual signal, is
    carried out to remove false positives occurring in creaky speech regions.
    
    Note that this method is slightly different from the standard SE-VQ
    algorithm as the mean based signal is calculated using a variable window
    length. 
    
    This is set using an f0 contour interpolated over unvoiced
    regions and heavily smoothed. This is particularly useful for speech
    involving large f0 excursions (i.e. very expressive speech).

    :param x:  speech signal (in samples)
    :param fs: sampling frequency (Hz)
    :param f0: f0 contour (optional), otherwise its computed  using the RAPT algorithm
    :returns: GCI Glottal closure instants (in samples)
    
    References:
          Kane, J., Gobl, C., (2013) `Evaluation of glottal closure instant 
          detection in a range of voice qualities', Speech Communication
          55(2), pp. 295-314.
    

    ORIGINAL FUNCTION WAS CODED BY JOHN KANE AT THE PHONETICS AND SPEECH LAB IN 
    TRINITY COLLEGE DUBLIN ON 2013.
    
    THE SEDREAMS FUNCTION WAS CODED BY THOMAS DRUGMAN OF THE UNIVERSITY OF MONS
   
    THE CODE WAS TRANSLATED TO PYTHON AND ADAPTED BY J. C. Vasquez-Correa
    AT PATTERN RECOGNITION LAB UNIVERSITY OF ERLANGEN NUREMBER- GERMANY
    AND UNIVERSTY OF ANTIOQUIA, COLOMBIA
    JCAMILO.VASQUEZ@UDEA.EDU.CO
    https//jcvasquezc.github.io


    """
    if f0 is None:
        f0 = []
    F0min=20
    F0max=500
    if len(f0)==0 or sum(f0)==0:
        size_stepS=0.01*fs
        voice_bias=-0.2
        x=x-np.mean(x)
        x=x/np.max(np.abs(x))
        data_audiof=np.asarray(x*(2**15), dtype=np.float32)
        f0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=F0min, max=F0max, voice_bias=voice_bias, otype='f0')


    F0nz=np.where(f0>0)[0]
    F0mean=np.median(f0[F0nz])
    VUV=np.zeros(len(f0))
    VUV[F0nz]=1
    if F0mean<70:
        print('Utterance likely to contain creak')
        F0mean=80

    # Interpolate f0 over unvoiced regions and heavily smooth the contour

    ptos=np.linspace(0,len(x),len(VUV))
    VUV_inter=np.interp(np.arange(len(x)), ptos, VUV)

    VUV_inter[np.where(VUV_inter>0.5)[0]]=1
    VUV_inter[np.where(VUV_inter<=0.5)[0]]=0

    f0_int, f0_samp=create_continuous_smooth_f0(f0,VUV,x)

    T0mean = fs/f0_samp
    winLen = 25 # window length in ms
    winShift = 5 # window shift in ms
    LPC_ord = int((fs/1000)+2) # LPC order
    Ncand=5 # Number of candidate GCI residual peaks to be considered in the dynamic programming
    trans_wgt=1 # Transition cost weight
    relAmp_wgt=0.3 # Local cost weight

    
    #Calculate LP-residual and extract N maxima per mean-based signal determined intervals

    res = GetLPCresidual(x,winLen*fs/1000,winShift*fs/1000,LPC_ord, VUV_inter) # Get LP residual

    MBS = get_MBS(x,fs,T0mean) # Extract mean based signal

    interval = get_MBS_GCI_intervals(MBS,fs,T0mean,F0max) # Define search intervals

    [GCI_N,GCI_relAmp] = search_res_interval_peaks(res,interval,Ncand, VUV_inter) # Find residual peaks

    if len(np.asarray(GCI_N).shape) > 1:
        GCI = RESON_dyProg_mat(GCI_relAmp,GCI_N,F0mean,x,fs,trans_wgt,relAmp_wgt, plots=False) # Do dynamic programming
    else:
        print("------------- warning -------------------, not enough pitch periods to reconstruct the residual and glottal signals")        
        GCI = None

    return GCI

def IAIF(x,fs,GCI):
    """
    Function to carry out iterative and adaptive inverse filtering (Alku et al 1992).
    
    :param x: speech signal (in samples)
    :param fs: sampling frequency (in Hz)
    :param GCI: Glottal closure instants (in samples)
    :returns: glottal flow derivative estimate
    

    Function Coded by John Kane @ The Phonetics and Speech Lab
    Trinity College Dublin, August 2012


    THE CODE WAS TRANSLATED TO PYTHON AND ADAPTED BY J. C. Vasquez-Correa
    AT PATTERN RECOGNITION LAB UNIVERSITY OF ERLANGEN NUREMBER- GERMANY
    AND UNIVERSTY OF ANTIOQUIA, COLOMBIA
    JCAMILO.VASQUEZ@UDEA.EDU.CO
    https//jcvasquezc.github.io
    """

    p=int(fs/1000)+2 # LPC order

    x_filt=x-np.mean(x)
    x_filt=x_filt/max(abs(x_filt))

    # ------------------------------------------------
    # emphasise high-frequencies of speech signal (LPC order 1) - PART 2 & 3
    ord_lpc1=1
    x_emph=calc_residual(x_filt,x_filt,ord_lpc1,GCI)

    # ------------------------------------------------
    # first estimation of the glottal source derivative - PART 4 & 5
    ord_lpc2=p
    residual1=calc_residual(x_filt,x_emph,ord_lpc2,GCI)

    # integration of the glottal source derivative to calculate the glottal
    # source pulse - PART 6 (cancelling lip radiation)
    ug1=cumtrapz(residual1)
    # ------------------------------------------------
    # elimination of the source effect from the speech spectrum - PART 7 & 8

    ord_lpc3=4
    vt_signal=calc_residual(x_filt,ug1,ord_lpc3,GCI)

    # ------------------------------------------------
    # second estimation of the glottal source signal - PART 9 & 10

    ord_lpc4=p
    residual2=calc_residual(x_filt,vt_signal,ord_lpc4,GCI)
    return residual2




def extract_glottal_signal(x, fs):
    """Extract the glottal flow and the glottal flow derivative signals

    :param x: data from the speech signal.
    :param fs: sampling frequency
    :returns: glottal signal
    :returns: derivative  of the glottal signal
    :returns: glottal closure instants

    >>> from scipy.io.wavfile import read
    >>> glottal=Glottal()
    >>> file_audio="../audios/001_a1_PCGITA.wav"
    >>> fs, data_audio=read(audio)
    >>> glottal, g_iaif, GCIs=glottal.extract_glottal_signal(data_audio, fs)

    """
    winlen=int(0.025*fs)
    winshift=int(0.005*fs)
    x=x-np.mean(x)
    x=x/float(np.max(np.abs(x)))
    GCIs=SE_VQ_varF0(x,fs)
    g_iaif=np.zeros(len(x))
    glottal=np.zeros(len(x))
    wins=np.zeros(len(x))

    if GCIs is None:
        print("------------- warning -------------------, not enought voiced segments were found to compute GCI")
        return glottal, g_iaif, GCIs

    start=0
    stop=int(start+winlen)
    win = np.hanning(winlen)

    while stop <= len(x):

        x_frame=x[start:stop]
        pGCIt=np.where((GCIs>start) & (GCIs<stop))[0]
        GCIt=GCIs[pGCIt]-start


        g_iaif_f=IAIF(x_frame,fs,GCIt)
        glottal_f=cumtrapz(g_iaif_f, dx=1/fs)
        glottal_f=np.hstack((glottal[start], glottal_f))
        g_iaif[start:stop]=g_iaif[start:stop]+g_iaif_f*win
        glottal[start:stop]=glottal[start:stop]+glottal_f*win
        start=start+winshift
        stop=start+winlen
    g_iaif=g_iaif-np.mean(g_iaif)
    g_iaif=g_iaif/max(abs(g_iaif))

    glottal=glottal-np.mean(glottal)
    glottal=glottal/max(abs(glottal))
    glottal=glottal-np.mean(glottal)
    glottal=glottal/max(abs(glottal))

    return glottal, g_iaif, GCIs