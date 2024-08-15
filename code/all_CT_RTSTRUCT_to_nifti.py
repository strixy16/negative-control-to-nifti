import pandas as pd
import os
from imgtools.io import read_dicom_series
from readii.loaders import loadSegmentation
from readii.image_processing import alignImages
from readii.negative_controls import applyNegativeControl
from joblib import Parallel, delayed
import SimpleITK as sitk
from argparse import ArgumentParser
import yaml

def parser():
    parser = ArgumentParser()

    parser.add_argument("config_file", type=str,
                        help="Path to YAML file with settings for the run")
    
    return parser.parse_known_args()[0]


def niftiConversion(edgeGraphPath, imageDirPath, outputDir, roiNames, controlTypes, controlRegions, randomSeed):

    # load edge file
    edgeGraphPath = "/Users/katyscott/Documents/RADCURE/.imgtools/imgtools_RADCURE_edges.csv"
    outputDir = "/Users/katyscott/Documents/HNC Project/scripts/negative-control-to-nifti/nc_niftis/"
    imageDirPath = "/Users/katyscott/Documents/RADCURE/"
    roiNames = ["GTVp*"]
    randomSeed = 10

    dfEdgeGraph = pd.read_csv(edgeGraphPath)

    dfCTWithRTSTRUCTS = dfEdgeGraph.loc[dfEdgeGraph['edge_type'] == 2]
    ctSeriesIDList = dfCTWithRTSTRUCTS["series_x"].unique()
    

    if not os.path.exists(outputDir):
        print("Creating output directory:", outputDir)
        os.makedirs(outputDir)
        
    def ctProcessing(ctSeriesID):
        ctSeriesInfo = dfCTWithRTSTRUCTS.loc[dfCTWithRTSTRUCTS["series_x"] == ctSeriesID]
        patID = ctSeriesInfo.iloc[0]["patient_ID_x"]
        print("Processing ", patID, "...")

        patOutDir = os.path.join(outputDir, patID)
        if not os.path.exists(patOutDir):
            os.makedirs(patOutDir)

        ctOutDir = os.path.join(patOutDir, "CT")
        if not os.path.exists(ctOutDir):
            os.makedirs(ctOutDir)
        
        rtOutDir = os.path.join(patOutDir, "RTSTRUCT_CT")
        if not os.path.exists(rtOutDir):
            os.makedirs(rtOutDir)
        
        # load CT from DICOM
        ctDirPath = os.path.join(imageDirPath, ctSeriesInfo.iloc[0]["folder_x"])
        ctImage = read_dicom_series(path=ctDirPath, series_id=ctSeriesID)

        # Load RTSTRUCT from DICOM using regex
        segFilePath = os.path.join(imageDirPath, ctSeriesInfo.iloc[0]["file_path_y"])
        segImages = loadSegmentation(segFilePath,
                                    modality=ctSeriesInfo.iloc[0]["modality_y"],
                                    baseImageDirPath=ctDirPath,
                                    roiNames=roiNames)
        # Get the first ROI mask
        roiLabel = list(segImages.keys())[0]
        roiMask = segImages[roiLabel]

        # Align the ROI mask to the CT image
        roiMask = alignImages(ctImage, roiMask)

        # Save CT to nifti - CT.nii.gz
        sitk.WriteImage(ctImage, os.path.join(ctOutDir, "CT.nii.gz"))
        # Save RTSTRUCT to nifti - GTVp.nii.gz
        sitk.WriteImage(roiMask, os.path.join(rtOutDir, "GTVp.nii.gz"))

        # Save each negative control to nifti - CT_negative_control_name.nii.gz
        for type in controlTypes:
            for region in controlRegions:
                print(type, region)
                outFileName = "CT_" + type + "_" + region + ".nii.gz"
                fullOutPath = os.path.join(ctOutDir, outFileName)
            
                ncCTImage = applyNegativeControl(baseImage=ctImage,
                                    negativeControlType = type,
                                    negativeControlRegion= region,
                                    roiMask = roiMask,
                                    randomSeed = randomSeed)
                
                sitk.WriteImage(ncCTImage, fullOutPath)

        return

    Parallel(n_jobs=-1, require="sharedmem")(delayed(ctProcessing)(ctSeriesID) for ctSeriesID in ctSeriesIDList)
    print("Pipeline complete")

    return

if __name__ == "__main__":
    args = parser()

    config = yaml.safe_load(open(args.config_file))

    niftiConversion(edgeGraphPath = config['edgeGraphPath'],
                    imageDirPath = config['imageDirPath'],
                    outputDir = config['outputDir'],
                    roiNames = config['roiNameRegex'],
                    controlTypes = config['controlTypes'],
                    controlRegions = config['controlRegions'],
                    randomSeed = config['randomSeed']
                    )