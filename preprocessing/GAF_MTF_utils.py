## This code was borrowed from the original authors of the
# GASDF/MTF paper github link:
# https://github.com/cauchyturing/Imaging-time-series-to-improve-classification-and-imputation

#PAA function
def paa(series, now, opw):
    if now == None:
        now = len(series) / opw
    if opw == None:
        opw = len(series) / now
    return [sum(series[i * opw : (i + 1) * opw]) / float(opw) for i in range(now)]

#Rescale data into [0,1]
def rescale(serie):
    maxval = max(serie)
    minval = min(serie)
    gap = float(maxval-minval)
    return [(each-minval)/gap for each in serie]

#Rescale data into [-1,1]
def rescaleminus(serie):
    maxval = max(serie)
    minval = min(serie)
    gap = float(maxval-minval)
    return [(each-minval)/gap*2-1 for each in serie]

#Generate quantile bins
def QMeq(series, Q):
    q, bins = pd.qcut(list(set(series)), Q, retbins=True)
    dic = dict(zip(set(series), q.codes))
    MSM = np.zeros([Q,Q])
    label = []
    for each in series:
        label.append(dic[each])
    for i in range(0, len(label)-1):
        MSM[label[i]][label[i+1]] += 1
    for i in xrange(Q):
        if sum(MSM[i][:]) == 0:
            continue
        MSM[i][:] = MSM[i][:]/sum(MSM[i][:])
    return np.array(MSM), label, bins

#Generate Markov Matrix given a spesicif number of quantile bins
def paaMarkovMatrix(paalist,level):
    paaindex = []
    for each in paalist:    
        for k in range(len(level) - 1):
            lower = float(level[k])
            upper = float(level[k + 1])
            if each >=lower and each <= upper:
                paaindex.append(k)
    return paaindex


