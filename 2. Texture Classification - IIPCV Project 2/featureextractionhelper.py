import numpy as np
import mahotas as mt
import mahotas.features.surf
import mahotas.features.texture
import pandas as pd
import skimage as sk
from skimage import feature
from IPython.display import display

class ExtractionHelper:
    @staticmethod
    def extract_features_haralick(image) -> np.ndarray:
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)
        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean
    @staticmethod
    def extract_features_LBP(image,degree=8,radius=8) -> np.ndarray: #-> np.float64:
        # calculate lbp
        textures = mt.features.lbp(image,degree,radius)
        # # take the mean of it and return it
        # ht_mean = textures.mean(axis=0)
        # return ht_mean
        return textures
    @staticmethod
    def extract_features_TAS(image) -> np.ndarray: #-> np.float64:
        # calculate lbp
        textures = mt.features.tas(image)
        # # take the mean of it and return it
        # ht_mean = textures.mean(axis=0)
        # return ht_mean
        return textures
    @staticmethod
    def extract_features_Zernike(image,degree=8,radius=8) -> np.ndarray: #-> np.float64:
        # calculate lbp
        textures = mt.features.zernike(image,degree,radius)
        # # take the mean of it and return it
        # ht_mean = textures.mean(axis=0)
        # return ht_mean
        return textures
    @staticmethod
    def extract_features_SURF(image) -> np.ndarray:
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.surf.surf(image)
        # take the mean of it and return it
        srf_mean = textures.mean(axis=0)
        return srf_mean
    @staticmethod
    def extract_features_GLCM(image) -> np.ndarray: #->pd.DataFrame:
        # FEATURES = ["GLCM_Contrast", "GLCM_Dissimilarity", "GLCM_Homogeneity", "GLCM_Energy", "GLCM_Correlation", "GLCM_ASM"]
        # featureMatrixGLCMDF = pd.DataFrame(columns=FEATURES)
        featureVec = getGLCMImageFeatures(image)
        # featureMatrixGLCMDF.loc[0]=featureVec
        # print(featureVec)
        # display(featureMatrixGLCMDF.head(10))
        # return featureMatrixGLCMDF.mean(axis=0)
        return np.array(featureVec)
    
    
    
# for imageNum in range(len(wholeDF)):

#     enhancedImage, morphedImage = getPreprocessedImage(wholeDF, imageNum)

#     # Our greyscale image is the enhancedImage and our binary thresholded image is the morphedImage
#     featureVector = getGLCMImageFeatures(enhancedImage, morphedImage, [0])

#     featureMatrixGLCMDF.loc[imageNum] = featureVector

# featureMatrixGLCMDF

def getGLCMImageFeatures(grayscaleImage) ->np.ndarray:
        featureVector = []
        
        graycom = feature.graycomatrix(grayscaleImage, [1], [0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
        featureVector.append(feature.graycoprops(graycom, 'contrast')[0, 0])
        featureVector.append(feature.graycoprops(graycom, 'dissimilarity')[0, 0])
        featureVector.append(feature.graycoprops(graycom, 'homogeneity')[0, 0])
        featureVector.append(feature.graycoprops(graycom, 'energy')[0, 0])
        featureVector.append(feature.graycoprops(graycom, 'correlation')[0, 0])
        featureVector.append(feature.graycoprops(graycom, 'ASM')[0, 0])
        return featureVector