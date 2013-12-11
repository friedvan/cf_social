import random
import numpy as np
import time

def load_dataset(filename):
    UO = {}
    n = 0
    for line in open(filename):
        n += 1
        if n % 1000000 == 0:
            print n
        col = line.rstrip().split('\t')
        uid = col[0]
        item = col[1]
                
        if uid not in UO:
            UO[uid] = {}
        UO[uid][item] = True
        
        idict[item] = True
        
    print 'load complete'
    return UO

def jaccard(user1, user2):
    intersection = 0

    len1 = len(user1)
    len2 = len(user2)
    #loop faster
    if len1 > len2:
        for u in user2:
            if u in user1:
                intersection += 1
    else:
        for u in user1:
            if u in user2:
                intersection += 1

    union = len1 + len2 - intersection
    if union != 0:
        return float(intersection) / union
    else:
        return 0.0

def similarity(u, dataset):
    sim = {}
    for uj in dataset:
        if uj == u:
            continue
        s = jaccard(dataset[u], dataset[uj])
        if s != 0.0:
            sim[uj] = s
    return sim

def combine(movie_sim, group_sim, alpha):
    for u in movie_sim:
        if u in group_sim:
##            movie_sim[u] = movie_sim[u] ** (1 - alpha) + group_sim[u] ** alpha
            movie_sim[u] = movie_sim[u] * (1 - alpha) + group_sim[u] * alpha
    

def user_similarity(u, train, group, alpha):
    if u not in M:
        movie_sim = similarity(u, train)
        M[u] = movie_sim
    else:
        movie_sim = M[u]
    if u not in G:
        group_sim = similarity(u, group)
        G[u] = group_sim
    else:
        group_sim = G[u]
        
    combine(movie_sim, group_sim, alpha)
    return movie_sim

def recommend(u, sim, train):
    reclist = {}
    for uj in sim:
        for i in train[uj]:
            if i in train[u]:
                continue
            reclist.setdefault(i, 0.0)
            reclist[i] += sim[uj]
    return reclist

def fill_reclist(rec, trainitem):
    srec = sorted(rec.items(), key=lambda a:a[1], reverse=True)
    reclist = []
    i=0
    for s in srec:
        reclist.append(s[0])
        i+=1
        if i >= RECLEN:
            break
        
        
    while len(reclist) < RECLEN:
        ran = random.randint(0, len(items) - 1)
        if items[ran] not in trainitem:
            reclist.append(items[ran])
    
    return reclist

def hitcount(like, reclist):
    hit = 0
    for i in reclist:
        if i in like:
            hit += 1
    return hit

def clear_dataset(dataset, udict):
    for u in dataset.keys():
        if u not in udict:
            del dataset[u]  
def user_knn(sim, k):
    knn = {}
    sknn = sorted(sim.items(), key=lambda a:a[1], reverse=True)
    if len(sknn) > k:
        sknn = sknn[:k]
    for u in sknn:
        knn[u[0]] = u[1]
    return knn

def sample_users(udict):
    users = []
    ud = udict.keys()
    idx = np.random.randint(0, len(ud), size=(NSAMPLE, 1))
    for i in idx:
        users.append(ud[i])
    return users
######################################################################

RECLEN = 20
NSAMPLE = 100
KNN = 200

idict = {}

train = load_dataset('trainset.txt')
test = load_dataset('testset.txt')
group = load_dataset('follow.txt')
#!!!
items = idict.keys()

udict  = dict([(u, True) for u in test if u in train and u in group])
clear_dataset(train, udict)
clear_dataset(test, udict)
clear_dataset(group, udict)

users = sample_users(udict)

M={}
G={}

n = 0
fout = open('out.txt', 'w')
for alpha in np.arange(0.0, 1.1, 0.1):
    p, recall = [], [0.0, 0]
    for u in users:
        n += 1
        t1=time.clock()
        sim = user_similarity(u, train, group, alpha)
        t2=time.clock()
        knn = user_knn(sim, KNN)
        t3 = time.clock()
        rec = recommend(u, knn, train)
        t4=time.clock()
        hit = hitcount(test[u], fill_reclist(rec, train[u]))
        t5=time.clock()
        cp = float(hit) / RECLEN
        p.append(cp)
        recall[0] += hit
        recall[1] += len(test[u])
    ##    print u, alpha, cp, float(hit) / len(test[u])
        line = ' '.join([str(a) for a in [sum(p)/float(len(p)), float(recall[0]) / recall[1]]])
        print n, u, t2-t1, t3-t2, t4-t3, t5-t4
    pr = '\t'.join([str(a) for a in [sum(p)/float(len(p)), float(recall[0]) / recall[1]]])
    lout = '\t'.join([str(alpha), pr])
    print lout
    fout.write(lout + '\n')
fout.close()















