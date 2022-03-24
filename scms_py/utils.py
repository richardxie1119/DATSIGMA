import pandas as pd
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pyimzml.ImzMLWriter import ImzMLWriter
from pyImagingMSpec.inMemoryIMS import inMemoryIMS
import os


def loadBrukerFIDs(file_path, fid_length, read_length, fid_idx, verbose = False):
    """

    """
    fids = []
    if os.path.exists(file_path):
        f = open(file_path,'r')

        if type(fid_idx) == list or type(fid_idx) == np.ndarray:
            #print('loading {} FID from file...'.format(len(fid_idx)))
            for i in range(len(fid_idx)):
                if verbose:
                    print('loading {} FID from file...'.format(i))

                f.seek(4*(fid_idx[i]-1)*fid_length) #seek FID locations within the .ser file

                if read_length == 'all':
                    fid = np.fromfile(f, count = fid_length, dtype = 'int32')
                    fids.append(fid)
                else:
                    fid = np.fromfile(f, count = read_length, dtype = 'int32')
                    fids.append(fid)
        else:
            f.seek(4*(fid_idx-1)*fid_length) #seek FID locations within the .ser file

            if read_length == 'all':
                fid = np.fromfile(f, count = fid_length, dtype = 'int32')
                fids.append(fid)
            else:
                fid = np.fromfile(f, count = read_length, dtype = 'int32')
                fids.append(fid)

        f.close()

    return np.array(fids,dtype='float64')



def parseBrukerMethod(file_path):

    tree = ET.parse(file_path)
    root = tree.getroot()

    for type_tag in root.findall('paramlist')[0]:
        name = type_tag.get('name')
        if name == 'SW_h':
            SW_h = float(type_tag.findall('value')[0].text)
        if name == 'TD':
            TD = int(type_tag.findall('value')[0].text)
        if name == 'ML1':
            ML1 = float(type_tag.findall('value')[0].text)
        if name == 'ML2':
            ML2 = float(type_tag.findall('value')[0].text)
        if name == 'ML3':
            ML3 = float(type_tag.findall('value')[0].text)

    return {'SW_h':SW_h,'TD':TD,'ML1':ML1,'ML2':ML2,'ML3':ML3}


def parseBrukerXML(file_path, detailed = False):
    
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    intensity = []
    mzs = []
    res = []
    snr = []
    
    for type_tag in root.findall('analysis/DataAnalysis/ms_spectrum/ms_peaks/'):
        intensity.append(float(type_tag.get('i')))
        mzs.append(float(type_tag.get('mz')))
        if detailed:
            res.append(float(type_tag.get('res')))
            snr.append(float(type_tag.get('sn')))
        
    return {'intensity':np.array(intensity),'mzs':np.array(mzs),'res':np.array(res),'snr':np.array(snr)}


def parseBrukerXML_tof(file_path, detailed = False):
    
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    intensity = []
    mzs = []
    res = []
    snr = []

    for type_tag in root.findall('pk'):
        intensity.append(float(type_tag.findall('absi')[0].text))
        mzs.append(float(type_tag.findall('mass')[0].text))
        if detailed:
            res.append(float(type_tag.findall('reso')[0].text))
            snr.append(float(type_tag.findall('s2n')[0].text))
        
    return {'intensity':np.array(intensity),'mzs':np.array(mzs),'res':np.array(res),'snr':np.array(snr)}


def loadmatfile(file_dir):
    
    f = loadmat(file_dir)
    names = f['data']['names'][0][0]
    name_list = [i[0] for i in names[0]]
    intens_matrix = f['data']['intens'][0][0].T
    mzs = f['data']['mzs'][0][0][0]

    print('Loaded intensity matrix with shape {}'.format(intens_matrix.shape))
    intens_df = pd.DataFrame(intens_matrix)
    intens_df = intens_df.set_index([pd.Index(name_list)])
    #intens_df[intens_df==0]=1
    #intens_df = np.log(intens_df)
    intens_df.columns = np.round(mzs,4)
    
    return intens_df

def pklist2imzML(peak_list, file_name, coords):

    """
    """
    keys = peak_list.keys()
    with ImzMLWriter(file_name+'.imzML') as w:

        idx = 0
        for key in keys:
            # writes data to the .ibd file
            #print(i)
            if len(peak_list[key]['mzs']) >0:
                w.addSpectrum(mzs = peak_list[key]['mzs'],intensities = peak_list[key]['intensity'],
                                        coords = coords[idx])
            idx += 1
    
    print('succefully parsed the peak list to imzml!')




