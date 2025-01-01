#creating the Gram-Schmidt Process (and some other linear algebra vector stuff) using functional programming techniques in Python

import math

def checkIfVectorsAreEqualDimension(v1,v2): #given vectors v1 and v2, raises an error if their dimension is not equal
    if len(v1) != len(v2):
        raise Exception("cannot perform cross product on vectors of different dimensions")

def isMatrixOrthogonal(M): #given a matrix M, returns whether M has orthogonal columns
    def isMatOrthHelper(v,series):
        if series == []:
            return True
        elif isOrthogonal(v,series[0]):
            return(isMatOrthHelper(v,series[1:]))
        else:
            return False
    if M == []:
        return True
    elif isMatOrthHelper(M[0],M[1:]):
        return isMatrixOrthogonal(M[1:])
    else:
        return False


def isOrthogonal(v1,v2): #given vectors v1 and v2, returns whether v1 and v2 are orthogonal
    isOrth = abs(dotProduct(v1,v2)) < 1e-5
    return isOrth

def vectorLength(v): #given a vector v, returns the length of v
    dotProd = dotProduct(v,v)
    length = math.sqrt(dotProd)
    return length

def normalize(v): #given a vector v, returns a normalized version of it (length 1)
    length = vectorLength(v)
    if length == 0:
        return v
    else:
        invLength = 1 / length
        return scalarMultiply(v,invLength)

def vectorSum(v1,v2): #given vectors v1 and v2, returns the sum of two vectors
    checkIfVectorsAreEqualDimension(v1,v2)
    
    def vectorSumHelper(vv1,vv2):
        if vv1 == [] or vv2 == []:
            return []
        else:
            return [vv1[0] + vv2[0]] + vectorSumHelper(vv1[1:],vv2[1:])

    return vectorSumHelper(v1,v2)

def scalarMultiply(vector,scalar): #given a vector and a scalar, returns the scalar multiple of the vector
    if vector == []:
        return []
    else:
        return [vector[0] * scalar] + scalarMultiply(vector[1:],scalar)

def dotProduct(v1,v2): #given vectors v1 and v2, returns the dot product of the two vectors
    checkIfVectorsAreEqualDimension(v1,v2)
    
    def dotProductHelper(vv1,vv2):
        if vv1 == [] or vv2 == []:
            return 0
        else:
            return vv1[0] * vv2[0] + dotProductHelper(vv1[1:],vv2[1:])

    return dotProductHelper(v1,v2)

def returnProjection(u, v): #given vectors u and v, returns the projection of the vector u onto v
    checkIfVectorsAreEqualDimension(u,v)
    udotv = dotProduct(u,v)
    vdotv = dotProduct(v,v)
    scale = udotv/vdotv
    proj = scalarMultiply(v,scale)
    return proj

def returnProjectionBasis(y,W): #given the vector y and the basis W, 
                                #returns the projection of the vector y onto the basis W
    if W == []:
        return [0] * len(y)
    else:
        return vectorSum(returnProjection(y,W[0]),returnProjectionBasis(y,W[1:]))
    
def orthogonalComplement(u,v): #give vectors u and v, returns the orthogonal 
                                #complement to the projection of u onto v
    checkIfVectorsAreEqualDimension(u,v)
    proj = returnProjection(u,v)
    negproj = scalarMultiply(proj,-1)
    comp = vectorSum(u,negproj)
    return comp

def orthogonalComplementBasis(y,W): #given a vector y and a basis W, returns the orthogonal compliment
                                    #of the projection of the vector y onto the basis W
    proj = returnProjectionBasis(y,W)
    negproj = scalarMultiply(proj,-1)
    comp = vectorSum(y,negproj)
    return comp

def normalizeMatrix(M): #given a matrix M, returns the normalized matrix M
    if M == []:
        return []
    else:
        return [normalize(M[0])] + normalizeMatrix(M[1:])

def GramSchmidtProcess(vectorBasis): #given a vector basis, returns a orthogonal matrix using the Gram-Schmidt Process
    def gspHelper(existingBasis,remainingBasis):
        if remainingBasis == []:
            return existingBasis
        else:
            #in Haskell we could use foldr/foldl using the orthogonalCompliment function instead of having a new function for a basis
            newOrtho = orthogonalComplementBasis(remainingBasis[0],existingBasis)
            newExistingBasis = existingBasis + [newOrtho]
            newRemainingBasis = remainingBasis[1:]
            return gspHelper(newExistingBasis,newRemainingBasis)
        
    return gspHelper([],vectorBasis)
