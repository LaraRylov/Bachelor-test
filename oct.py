import math
import numpy as np
import gurobipy as gp


from collections import namedtuple
from sklearn import tree
from gurobipy import GRB
from scipy import stats #bruges til stats.mode
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class OCT:

  def __init__(self, max_depth, min_samples_leaf, alpha, warmstart=True, timelimit=600, output = True):
    self.max_depth = max_depth
    self.min_samples_leaf = min_samples_leaf                              # minimum antal datapunkter i enhver slutnode
    self.alpha = alpha
    self.warmstart = warmstart
    self.output = output
    self.timelimit = timelimit
    self.trained = False
    self.optgap = None

#Indekserer over nodes
    self.index_nodes = [t+1 for t in range(2**(self.max_depth + 1) -1)]                                         # Nodes
    self.index_branch_nodes = [t+1 for t in range(math.floor(max(self.index_nodes)/2))]                         # Branch nodes
    self.index_leaf_nodes = [t+1 for t in range(math.floor(max(self.index_nodes)/2), max(self.index_nodes))]    # Leaf nodes


  def fit(self, x, y):
# Data size
    self.n, self.p = x.shape              #self.n er observationer, self.p er indgange, x.shape giver antal rækker/instances samt antal kollonner/features
    if self.output:
        print('Training data include {} instances, {} features.'.format(self.n,self.p)) #If true: Output = (observationer,indgange)

# Labels – giver alle vores klasser en gang
    self.labels = np.unique(y)

# Scale data???
    self.scales = np.max(x, axis=0)     #Finder den maksimale værdi for hver feature (kolonne)
    self.scales[self.scales == 0] = 1   #Hvis der findes maks-værdier på 0, erstattes de med 1 for at undgå divison med 0 senere


#Solve MIP
    m, a, b, c, d, z, l = self._buildMIP(x/self.scales, y)
    if self.warmstart:
          self._setStart(x, y, a, c, d, l)
    m.optimize()
    self.optgap = m.MIPGap # MIP-gapet er forskellen mellem den bedste løsning, der er fundet indtil videre, og den bedste mulige løsning.


#Get parameters
    self._a = {ind:a[ind].x for ind in a}       #Laver dictonaries så vi kan printe løsningerne
    self._b = {ind:b[ind].x for ind in b}
    self._c = {ind:c[ind].x for ind in c}
    self._d = {ind:d[ind].x for ind in d}
    self._z = {ind:z[ind].x for ind in z}
    self._l = {ind:l[ind].x for ind in l}

    self.trained = True


  def predict(self, x):

        if not self.trained:
            raise AssertionError('This optimalDecisionTreeClassifier instance is not fitted yet.') #hvis ikke modellen er fitted, kommer denne kommentar

# leaf label
        labelmap = {}                                 #Laver tom dictonary til label predicting
        for t in self.index_leaf_nodes:               #for hver leaf node t
            for k in self.labels:                     #for hver klasse-label k
                if self._c[k,t] >= 1e-2:              #hvis c_kt>0 (altså hvis c_kt=1) så tildeles node t label k
                    labelmap[t] = k

        y_pred = []                                   #Laver tom liste der skal indeholde predicted class labels for hver datapunkt
        for xi in x/self.scales:                      #gennemløber nu alle datapunkter
            t = 1                                     #starter fra root node
            while t not in self.index_leaf_nodes:              #for t ikke en leaf node
                right = (sum([self._a[j,t] * xi[j] for j in range(self.p)]) + 1e-9 >= self._b[t])
                if right:
                    t = 2 * t + 1                     #hvis datapunkt x_1 følger den højre gren, sættes t til indekset på child node'en (ulige)
                else:
                    t = 2 * t                         #ellers sættes t til indekset på child node'en til venstre (lige)

                    # label
            y_pred.append(labelmap[t])                #når den når til en node t der er en leaf node, tildeles x_i den forudsagte klasse

        return np.array(y_pred)



  def _buildMIP(self, x, y):
    m = gp.Model()                            #Laver modellen m


    m.Params.outputFlag = self.output               #sørger for at der printes output (om optimeringen)
    m.Params.LogToConsole = self.output             #kontrolerer om logging information fra solveren vises
    # time limit
    m.Params.timelimit = self.timelimit
    # parallel
    m.params.threads = 0                            #tillader solveren at bestemme det optimale antal threads. Maksimerer performance
    #m.params.MIPFocus=2



    m.modelSense = GRB.MINIMIZE                     #Minimerer objektfunktionen


# Variables
    a = m.addVars(self.p, self.index_branch_nodes, vtype=GRB.BINARY, name='a')        # splitting feature: over indgang, branchnodes
    d = m.addVars(self.index_branch_nodes, vtype=GRB.BINARY, name='d')                # giver 1, hvis node t splitter og 0 hvis ikke
    b = m.addVars(self.index_branch_nodes, vtype=GRB.CONTINUOUS, name='b')            # højresiden, når vi splitter
    z = m.addVars(self.n, self.index_leaf_nodes, vtype=GRB.BINARY, name='z')          # leaf node assignment
    l = m.addVars(self.index_leaf_nodes, vtype=GRB.BINARY, name='l')                  # giver 1, hvis der er mindst et punkt i bladet og 0 ellers
    N = m.addVars(self.labels, self.index_leaf_nodes, vtype=GRB.CONTINUOUS, name='Nkt') # antal punkter med klasse k i blad t
    Nt = m. addVars(self.index_leaf_nodes, vtype=GRB.CONTINUOUS, name='Nt')             # samlet antal punkter i node t
    c = m.addVars(self.labels, self.index_leaf_nodes, vtype=GRB.BINARY, name='c')     # giver 1, hvis label k er givet til blad t og 0 ellers
    L = m.addVars(self.index_leaf_nodes, vtype=GRB.CONTINUOUS, name='l')              # antal datapunkter misklassificeret i blad t



    # calculate baseline accuracy
    Lhat = self._calLhat(y)                      # beregner baseline-værdien på klasse y


    # epsilon
    epsilon = self._epsilon(x)

    # objektfunktion
    obj = L.sum() / Lhat + self.alpha * d.sum()
    m.setObjective(obj)

# Bibetingelser
    #(2.1)
    m.addConstrs(a.sum('*', t) == d[t] for t in self.index_branch_nodes) #summen af a_jt lig med d_t for hver branch nodes t
    #(2.2)
    m.addConstrs(b[t] <= d[t] for t in self.index_branch_nodes) # 0 <= b_t <= d_t for alle branch nodes t
    #(2.4)
    m.addConstrs(d[t] <= d[t//2] for t in self.index_branch_nodes if t != 1) #d_t <= d_p(t) for alle branch nodes undtagen t=1 (root node)
                                                                             # t//2 giver t/2 og så runder den ned til nærmeste heltal
    #(2.5)
    m.addConstrs(z[i,t] <= l[t] for t in self.index_leaf_nodes for i in range(self.n)) #z_it <= l_t for alle leaf nodes t og for alle i=1,..,n
    #(2.6)
    m.addConstrs(z.sum('*', t) >= self.min_samples_leaf * l[t] for t in self.index_leaf_nodes) #summen af z_it >= N_min * l_t for alle leaf nodes t
    #(2.7)
    m.addConstrs(z.sum(i, '*') == 1 for i in range(self.n)) #summen af z_it (over alle leaf nodes t) skal være lig med 1 for alle i=1,...,N (hvert punkt tildeles 1 leaf node)
    #(2.15) og (2.16)
    for t in self.index_leaf_nodes:
            left = (t % 2 == 0)           #Tager alle nodes med et lige indeks, det vil sige alle nodes til venstre (% betyder, at restleddet efter division er 0)
            t_anc = t // 2                #Tager parent node (ancestor)
            while t_anc != 0:             #Kigger ikke på root node, fordi 1//2 = 0
                if left:
                    m.addConstrs(gp.quicksum(a[j,t_anc] * (x[i,j] + epsilon[j]) for j in range(self.p))
                                 +
                                 (1 + np.max(epsilon)) * (1 - d[t_anc]) #Sikkerhedsforanstaltning??
                                 <=
                                 b[t_anc] + (1 + np.max(epsilon)) * (1 - z[i,t])
                                 for i in range(self.n))
                else:
                    m.addConstrs(gp.quicksum(a[j,t_anc] * x[i,j] for j in range(self.p))
                                 >=
                                 b[t_anc] - (1 - z[i,t])
                                 for i in range(self.n))
                left = (t_anc % 2 == 0)   #Tjekker igen om noden har et lige indeks
                t_anc //= 2               #Tager nu parent node til den node man netop har arbejdet på. Trævler sig op igennem træet på den måde
                                          # = gør at t_anc defineres på ny
    #(2.17)
    m.addConstrs(gp.quicksum((y[i] == k) * z[i,t] for i in range(self.n)) == N[k,t] for t in self.index_leaf_nodes for k in self.labels)
    #Summerer alle datapunkter med label k (y_ik) hvis datapunkt i er i leaf node t. Finder samlet antal datapunkter med label k i leaf node t
    #(2.19)
    m.addConstrs(z.sum('*', t) == Nt[t] for t in self.index_leaf_nodes) #Samlet antal punkter i leaf node t er lig med summen af z_it over i
    #(2.21)
    m.addConstrs(c.sum('*', t) == l[t] for t in self.index_leaf_nodes) #node t får kun et label k hvis den indeholder punkter, altså hvis l_t=1, ellers c_kt=0=l_t

    #For (2.23) og (2.24) er N_t=Nt[t], N_kt=N[k,t] og M=n=antal datapunkter (her self.n). Dette er lineariseringsbetingelserne
    #(2.23)
    m.addConstrs(L[t] >= Nt[t] - N[k,t] - self.n * (1 - c[k,t]) for t in self.index_leaf_nodes for k in self.labels)
    #(2.24)
    m.addConstrs(L[t] <= Nt[t] - N[k,t] + self.n * c[k,t] for t in self.index_leaf_nodes for k in self.labels)

    return m, a, b, c, d, z, l

  @staticmethod
  def _calLhat(y):
    mode = stats.mode(y)[0]                         # vælger den klasse, der går hyppigst igen i datasættet
    return np.sum(y == mode)                       # beregner baseline-værdien på klasse y



  def _epsilon(self,x):
    epsilon = []
    for j in range(x.shape[1]):
        xj = x[:,j]                                 # løber igennem alle observationer for et j
        xj = np.unique(xj)                          # fjerner de duplikerede værdier af xj, da epsilon ikke må blive 0
        xj = np.sort(xj)[::-1]                      # den rangerer værdierne fra højeste til laveste
        dis = [1]
        for i in range(len(xj)-1):                  # for x_j gennemløbes alle observationer (over i)
          dis.append(xj[i] - xj[i+1])               # den laver en liste med alle afstandene, dvs. xj[1]-xj[2], xj[2]-xj[3]
        epsilon.append(np.min(dis) if np.min(dis) else 1)
    return epsilon


  def _setStart(self, x, y, a, c, d, l):
        """
        set warm start from CART
        """
        # train with CART
        if self.min_samples_leaf > 1:
            clf = tree.DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        else:
            clf = tree.DecisionTreeClassifier(max_depth=self.max_depth)
        clf.fit(x, y)

        # get splitting rules
        rules = self._getRules(clf)               #henter kriterier for split fra getRules

        # fix branch node
        for t in self.index_branch_nodes:
                                                  #Hvis den ikke splitter
            if rules[t].feat is None or rules[t].feat == tree._tree.TREE_UNDEFINED:
                d[t].start = 0                    #hvis der ikke splittes sættes d_t=0
                for j in range(self.p):
                    a[j,t].start = 0
            # split
            else:
                d[t].start = 1                    #ellers sættes d_t=1
                for j in range(self.p):
                    if j == int(rules[t].feat):
                        a[j,t].start = 1          #a_jt sættes til 1 hvis der splittes på feature j
                    else:
                        a[j,t].start = 0          #a_jt sættes til 0 for alle j der ikke angiver et split

        # fix leaf nodes
        for t in self.index_leaf_nodes:           #løber over alle leaf nodes t
            # terminate early
            if rules[t].value is None:            #Hvis værdien er None er det en leaf node (brancher ikke)
                l[t].start = int(t % 2)
                # flows go to right
                if t % 2:
                    t_leaf = t
                    while rules[t].value is None:     #hvis None, sættes l(t)=0 for lige t og l(t)=1 for ulige t
                        t //= 2
                    for k in self.labels:
                        if k == np.argmax(rules[t].value):
                            c[k, t_leaf].start = 1     #Når l(t) for leaf node t ikke er none, er der datapunkter i leaf node t og den tildeles et label
                        else:
                            c[k, t_leaf].start = 0
                # nothing in left
                else:
                    for k in self.labels:
                        c[k, t].start = 0
            # terminate at leaf node
            else:
                l[t].start = 1
                for k in self.labels:
                    if k == np.argmax(rules[t].value):
                        c[k, t].start = 1
                    else:
                        c[k, t].start = 0

  def _getRules(self, clf):
        """
        get splitting rules
        """
        # node index map
        node_map = {1:0}
        for t in self.index_branch_nodes:
            # terminal
            node_map[2*t] = -1
            node_map[2*t+1] = -1
            # left
            l = clf.tree_.children_left[node_map[t]]
            node_map[2*t] = l
            # right
            r = clf.tree_.children_right[node_map[t]]
            node_map[2*t+1] = r


  # rules
        rule = namedtuple('Rules', ('feat', 'threshold', 'value'))
        rules = {}
        # branch nodes
        for t in self.index_branch_nodes:
            i = node_map[t]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(clf.tree_.feature[i], clf.tree_.threshold[i], clf.tree_.value[i,0])
            rules[t] = r
        # leaf nodes
        for t in self.index_leaf_nodes:
            i = node_map[t]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(None, None, clf.tree_.value[i,0])
            rules[t] = r

        return rules

  def print_solution(self):
      for j in range(self.p):
        for t in self.index_branch_nodes:
          print('a',j,t,'=',self.a[j,t].x)


  def print_solution(self):
    a_var = []
    for t in self.index_branch_nodes:
        for j in range(self.p):
            if self._a[j,t] == 1:
               a_var[t] = 1

    return a_var
