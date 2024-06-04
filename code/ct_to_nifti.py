from imgtools.io import read_dicom_series
from readii.loaders import loadSegmentation
from readii.image_processing import *
from readii.negative_controls import *
from readii.metadata import *
import SimpleITK as sitk
from joblib import Parallel, delayed
from typing import Optional
import os
import re
import time
import datetime
import yaml
from argparse import ArgumentParser


def parser():
    parser = ArgumentParser()

    parser.add_argument("config_file", type=str,
                        help="Path to YAML file with settings for the run")
    
    return parser.parse_known_args()[0]


def ctToNIFTI(ctImage, outputDir):
    try:
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        
        ctNiftiOutPath = os.path.join(outputDir, "original_ct.nii.gz")

        sitk.WriteImage(ctImage, ctNiftiOutPath)
         
    except Exception as e:
        print(str(e))
        print("CT to NIFTI conversion failed. Skipping patient, please review.")
    
    return


def main(imageDirPath, outputDir, imageFileListPath, segType):

    if not os.path.exists(outputDir):
            print("Creating output directory:", outputDir)
            os.makedirs(outputDir)

    pdImageInfo = matchCTtoSegmentation(imageFileListPath, segType)

    ctSeriesIDList = pdImageInfo["series_CT"].unique()

    for ctSeriesID in ctSeriesIDList:
        ctSeriesInfo = pdImageInfo.loc[pdImageInfo["series_CT"] == ctSeriesID]
        patID = ctSeriesInfo.iloc[0]["patient_ID"]
        print("Processing ", patID, "...")
        
        # Get absolute path to CT image files
        ctDirPath = os.path.join(imageDirPath, ctSeriesInfo.iloc[0]["folder_CT"])

        try:
            # Load CT by passing in specific series to find in a directory
            ctImage = read_dicom_series(path=ctDirPath, series_id=ctSeriesID)

            # find if patient dir already exists
            pattern = patID + "*"

            for filename in os.listdir(outputDir):
                if re.search(pattern, filename):
                    print("Found ", patID, " folder. Saving CT to NIFTI")
                    niftiOutDirPath = os.path.join(outputDir, filename)

                    ctToNIFTI(ctImage, niftiOutDirPath)


        except Exception as e:
            print(str(e))
            print("Processing error occurred for ", patID, ". Could not load CT. Skipping patient, please review.")
            continue


if __name__ == "__main__":
    args = parser()
    
    config = yaml.safe_load(open(args.config_file))

    main(imageDirPath = config['imageDirPath'], 
         outputDir = config['outputDir'], 
         imageFileListPath = config['imageFileListPath'],
         segType = config['segType'])