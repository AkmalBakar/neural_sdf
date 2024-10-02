import numpy as np
from decimal import *

class PointSampler(): 
    """Sample points close to triangle mesh surface and in it's bounding box"""
    def __init__(self, vertices, faces, bounding_box, ratio = 0.0, std=0.0):
        self._V = vertices
        self._F = faces
        self._BB = bounding_box

        if ratio < 0 or ratio > 1:
            raise(ValueError("Ratio must be [0,1]"))
        
        self._ratio = ratio

        if std < 0:
            raise(ValueError("Std deviation must be non-negative"))

        self._std = std

        self._calculateFaceBins()
    
    def _calculateFaceBins(self):
        """Calculates and saves face area bins for sampling against"""
        vc = np.cross(
            self._V[self._F[:, 0], :] - self._V[self._F[:, 2], :],
            self._V[self._F[:, 1], :] - self._V[self._F[:, 2], :])

        A = np.sqrt(np.sum(vc ** 2, 1))
        FA = A / np.sum(A)
        self._faceBins = np.concatenate(([0],np.cumsum(FA))) 

    def _surfaceSamples(self,n):
        """Returns n points uniformly sampled from surface of mesh"""
        R = np.random.rand(n)   #generate number between [0,1]
        sampleFaceIdxs = np.array(np.digitize(R,self._faceBins)) -1

        #barycentric coordinates for each face for each sample :)
        #random point within face for each sample
        r = np.random.rand(n, 2)
        A = self._V[self._F[sampleFaceIdxs, 0], :]
        B = self._V[self._F[sampleFaceIdxs, 1], :]
        C = self._V[self._F[sampleFaceIdxs, 2], :]
        P = (1 - np.sqrt(r[:,0:1])) * A \
                + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B \
                + np.sqrt(r[:,0:1]) * r[:,1:] * C

        return P

    def _verticeSamples(self, n):
        """Returns n random vertices of mesh"""
        verts = np.random.choice(len(self._V), n)
        return self._V[verts]
    
    def _normalDist(self, V):
        """Returns normal distribution about each point V"""
        if self._std > 0.0:
            return np.random.normal(loc = V,scale = self._std)

        return V
        
    def _randomSamples(self, n):
        """Returns n random points in bounding box"""
        return 

    def sample(self,n):
        """Returns n points according to point sampler settings"""

        nRandom = round(Decimal(n)*Decimal(self._ratio))
        nSurface = n - nRandom

        xRandom = self._randomSamples(nRandom)

        if nSurface > 0:
            xSurface = self._surfaceSamples(nSurface)

            xSurface = self._normalDist(xSurface)
            if nRandom > 0:
                x = np.concatenate((xSurface,xRandom))
            else:
                x = xSurface
        else:
            x = xRandom

        np.random.shuffle(x)    #remove bias on order

        return x