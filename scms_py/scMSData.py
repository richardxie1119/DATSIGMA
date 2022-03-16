import sys
import json
from tqdm import tqdm
import pickle
import os
import pandas as pd

sys.path.append('../')
from utils import *
from processing import *
import numpy as np
import random
from joblib import Parallel, delayed
from pyimzml.ImzMLWriter import ImzMLWriter
from pyImagingMSpec.inMemoryIMS import inMemoryIMS


class scMSData():

    """
    """

    def __init__(self):

        self.parameters = {}    #load parameter file stored as .json
        self.spectra = {}       #raw spectra
        self.coords = []        #cell x,y coordinates, e.g. tuple(71232, 10321)
        self.random_state = 19
        self.use_index = []


    def getXMLPath(self, path):

        file_paths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".xml"):
                     file_paths.append(os.path.join(root, file))

        self.file_paths = file_paths
        self.names = [file_path.split('/')[-1].split('.')[0] for file_path in file_paths]

        return file_paths


    def loadXMLData(self):

        self.peak_list = {}    #list of centroided peak features for the spectra

        for i in tqdm(range(len(self.file_paths))):
            self.peak_list[self.names[i]] = parseBrukerXML(self.file_paths[i])

    def loadXMLData_tof(self):

        self.peak_list = {}    #list of centroided peak features for the spectra

        for i in tqdm(range(len(self.file_paths))):
            self.peak_list[self.names[i]] = parseBrukerXML_tof(self.file_paths[i])


    def getIntensMtxData(self, mz_range, feature_n, ppm=2, from_imzml=True, mz_features=[], intens_array=np.array([])):

        if from_imzml:

            if len(mz_features) == 0:
                intens_array, mz_bins_use, c = extractMZFeatures(self.imzML_dataset, ppm, mz_range, feature_n=feature_n)
            else:
                intens_array, mz_bins_use, c = extractMZFeatures(self.imzML_dataset, ppm, mz_range, feature_n=feature_n, mz_bins=mz_features)

            intens_mtx = pd.DataFrame(intens_array, columns=mz_bins_use, index=self.names)

        else:
            intens_mtx = pd.DataFrame(intens_array, columns=mz_features, index=self.names)

        self.intens_mtx = intens_mtx


    def loadimzMLData(self, file_name):

        self.imzML_dataset = inMemoryIMS(file_name)

    def convertPeak2imzML(self, file_name):

        coords = [tuple([1,i]) for i in range(len(self.names))]
        pklist2imzML(self.peak_list, file_name, coords)

        self.loadimzMLData(file_name+'.imzML')


    def experimentInfo(self, file_dict_path):

            """
            """
            with open(file_dict_path, 'r') as fp:
                path_dict = json.load(fp)

            self.loadParams(path_dict['parameter_file_path'])       #load data parameters
            self.imaginginfo_HR = parseImagingInfo(path_dict['imaging_info_path_HR'])     #load imaging experiment information
            self.ser_file_path_HR = path_dict['ser_file_path_HR']

            
    def loadParams(self, parameter_file_path):

        """
        """

        with open(parameter_file_path, 'r') as fp:
            self.parameters = json.load(fp)

        self.parameters['T'] = 1/self.parameters['sw_h']
        self.parameters['t'] = np.arange(0, self.parameters['fid_length_HR'])*self.parameters['T']
        self.parameters['f'] = self.parameters['sw_h']*np.arange(0, self.parameters['fid_length_HR']/2+1)/self.parameters['fid_length_HR']
        self.parameters['m'] = fticr_mass_axis(self.parameters['f_HR'], self.parameters['CALIB'])


        print('loaded parameters for the experiment...')
        print(self.parameters)





