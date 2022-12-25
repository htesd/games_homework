from functools import reduce

import numpy as np
import pandas as pd


# Function to create a conditional probability table
# Conditional probability is of the form p(x1 | x2, ..., xk)
# varnames: vector of variable names (strings) first variable listed
#           will be x_i, remainder will be parents of x_i, p1, ..., pk
# probs: vector of probabilities for the flattened probability table
# outcomesList: a list containing a vector of outcomes for each variable
# factorTable is in the type of pandas dataframe
# See the example file for examples of how this function works

def readFactorTable(varnames, probs, outcomesList):
    factorTable = pd.DataFrame({'probs': probs})

    totalfactorTableLength = len(probs)
    numVars = len(varnames)

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(totalfactorTableLength / (k * numLevs))
        k = k * numLevs

    return factorTable


# Build a factorTable from a data frame using frequencies
# from a data frame of data to generate the probabilities.
# data: data frame read using pandas read_csv
# varnames: specify what variables you want to read from the table
# factorTable is in the type of pandas dataframe
def readFactorTablefromData(data, varnames):
    numVars = len(varnames)
    outcomesList = []

    for i in range(0, numVars):
        name = varnames[i]
        outcomesList = outcomesList + [list(set(data[name]))]

    lengths = list(map(lambda x: len(x), outcomesList))
    m = reduce(lambda x, y: x * y, lengths)

    factorTable = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(outcomesList[0])

    # creates the vector called fact to index probabilities 
    # using matrix multiplication with the data frame
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(outcomesList[i])

    # Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    for i in range(0, m):
        factorTable.at[i, 'probs'] = sum(a == (i + 1))

    # normalize the conditional probabilities
    skip = int(m / numLevels)
    for i in range(0, skip):
        normalizeZ = 0
        for j in range(i, m, skip):
            normalizeZ = normalizeZ + factorTable['probs'][j]
        for j in range(i, m, skip):
            if normalizeZ != 0:
                factorTable.at[j, 'probs'] = factorTable['probs'][j] / normalizeZ

    return factorTable


# Join of two factors
# factor1, factor2: two factor tables
#
# Should return a factor table that is the join of factor 1 and 2.
# You can assume that the join of two factors is a valid operation.
# Hint: You can look up pd.merge for mergin two factors
def joinFactors(factor1, factor2):
    f1 = pd.DataFrame.copy(factor1)
    f2 = pd.DataFrame.copy(factor2)

    #这里只能假设传进来的数值是对的，即相同的部分都是条件，不同的部分都是情况。然后我需要设计一个


    joinFactor = None
    # TODO: start your code
    # print(pd.concat([f1,f2]))
    #print("开始join")
    def exist_in_list(list, val):

        for i in list:

            if i == val:
                return True

        return False

    same_index = ['key']

    for i in f1.columns:

        for j in f2.columns:

            if (i == j) & (i != 'probs'):
                same_index.append(j)

    # 开始对不同的元素排列组合
    #他妈的sql白学了，只要多加一列就能指定怎么merge了他妈的

    f1 = f1.assign(key=1)

    f2 = f2.assign(key=1)

    # print(same_index)
    temp = f1.merge(f2, on=same_index, how='inner').drop('key', 1)
    # print(f1.merge(f2,on=same_index,how='inner').drop('key',1))
    # print("????????????????????")
    temp = temp.assign(probs=temp['probs_x'] * temp['probs_y'])

    temp = temp.drop('probs_x', axis=1)

    temp = temp.drop('probs_y', axis=1)

    #print(temp)

    joinFactor=temp
    # end of your code
    #print(joinFactor)
    return joinFactor


# Marginalize a variable from a factor
# table: a factor table in dataframe
# hiddenVar: a string of the hidden variable name to be marginalized
#
# Should return a factor table that marginalizes margVar out of it.
# Assume that hiddenVar is on the left side of the conditional.
# Hint: you can look can pd.groupby
def marginalizeFactor(factorTable, hiddenVar):
    factor = pd.DataFrame.copy(factorTable)

    if hiddenVar not in list(factor.columns):
        return factor

    # TODO: start your code
    #
    # print("开始margin")
    #
    # #print(factor)
    # temp1=factor.groupby(hiddenVar).get_group(1)
    # temp0=factor.groupby(hiddenVar).get_group(0)
    #
    # temp0.loc[:,'probs']+=temp1['probs'].to_numpy()
    #
    #
    # index_list=[]
    # for i in range(len(temp1['probs'])):
    #     index_list.append(i)
    #
    #
    #
    # temp0.index=index_list
    #
    #
    # temp0=temp0.drop(hiddenVar,1)
    #
    # factor=temp0
    #
    # print(factor)

    #我是傻逼，最后groupby求和就行了，不用指定
    var_list = [value for value in factor.columns if value != hiddenVar]
    var_list.remove('probs')

    factor = factor.groupby(var_list).sum()

    factor = factor.drop(columns=hiddenVar).reset_index()

    last_col = factor.pop(factor.columns[-1])
    factor.insert(0, last_col.name, last_col)
    # end of your code

    return factor


# Marginalize a list of variables
# bayesnet: a list of factor tables and each table in dataframe type
# hiddenVar: a string of the variable name to be marginalized
#
# Should return a Bayesian network containing a list of factor tables that results
# when the list of variables in hiddenVar is marginalized out of bayesnet.
def marginalizeNetworkVariables(bayesNet, hiddenVar):
    '''
    本质上来说，我们只需要一个全概率分布，所以一股脑的join起来，如果有hidden，那么把他消除就行了

    :param bayesNet:
    :param hiddenVar:
    :return:
    '''
    if isinstance(hiddenVar, str):
        hiddenVar = [hiddenVar]

    if not bayesNet or not hiddenVar:
        return bayesNet

    marginalizeBayesNet = bayesNet.copy()

    # TODO: start your code

    for var in hiddenVar:
        temp = []
        tempFactor = None
        for factor in marginalizeBayesNet:

            if var in factor.columns:

                if tempFactor is None:
                    tempFactor = factor
                else:
                    tempFactor = joinFactors(tempFactor, factor)
            else:
                temp.append(factor)

        if tempFactor is not None:
            temp.append(marginalizeFactor(tempFactor, var))

        marginalizeBayesNet = temp.copy()

    # end of your code

    return marginalizeBayesNet


# Update BayesNet for a set of evidence variables
# bayesNet: a list of factor and factor tables in dataframe format
# evidenceVars: a vector of variable names in the evidence list
# evidenceVals: a vector of values for corresponding variables (in the same order)
#
# Set the values of the evidence variables. Other values for the variables
# should be removed from the tables. You do not need to normalize the factors
def evidenceUpdateNet(bayesNet, evidenceVars, evidenceVals):

    if isinstance(evidenceVars, str):
        evidenceVars = [evidenceVars]
    if isinstance(evidenceVals, str):
        evidenceVals = [evidenceVals]

    updatedBayesNet = bayesNet.copy()
    # TODO: start your code

    #print("开始更新证据变量")

    # print(bayesNet)
    # print(evidenceVars)
    # print(evidenceVals)
    '''
    只保留和证据相同的元素，这pandas是真的难用
    '''
    result = []
    var_num = 0
    for factor in updatedBayesNet:
        for var in evidenceVars:
            if var in factor.columns:
                factor = factor[factor[var].isin([int(evidenceVals[var_num])])]

            var_num += 1
        result += [factor]
        var_num = 0
    updatedBayesNet = result
    # end of your code
    # print("???????")
    # print(updatedBayesNet)
    return updatedBayesNet


# Run inference on a Bayesian network
# bayesNet: a list of factor tables and each table iin dataframe type
# hiddenVar: a string of the variable name to be marginalized
# evidenceVars: a vector of variable names in the evidence list
# evidenceVals: a vector of values for corresponding variables (in the same order)
#
# This function should run variable elimination algorithm by using
# join and marginalization of the sets of variables.
# The order of the elimiation can follow hiddenVar ordering
# It should return a single joint probability table. The
# variables that are hidden should not appear in the table. The variables
# that are evidence variable should appear in the table, but only with the single
# evidence value. The variables that are not marginalized or evidence should
# appear in the table with all of their possible values. The probabilities
# should be normalized to sum to one.
def inference(bayesNet, hiddenVar, evidenceVars, evidenceVals):
    if not bayesNet:
        return bayesNet

    inferenceNet = bayesNet.copy()
    factor = None
    # TODO: start your code

    inferenceNet = evidenceUpdateNet(inferenceNet, evidenceVars, evidenceVals)

    inferenceNet = marginalizeNetworkVariables(inferenceNet, hiddenVar=hiddenVar)

    length = len(inferenceNet)
    if length == 1:
        factor = inferenceNet[0]
    else:
        factor = inferenceNet[0]
        for idx in range(1, length):
            factor = joinFactors(factor, inferenceNet[idx])
    # normalize
    norm = sum(list(factor['probs']))
    factor['probs'] /= norm


    return factor
