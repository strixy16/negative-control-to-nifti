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

    dfEdgeGraph = pd.read_csv(edgeGraphPath)
    dfCTWithRTSTRUCTS = dfEdgeGraph.loc[dfEdgeGraph['edge_type'] == 2]
    ctSeriesIDList = dfCTWithRTSTRUCTS["series_x"].unique()

    if not os.path.exists(outputDir):
        print("Creating output directory:", outputDir)
        os.makedirs(outputDir)
        
    def ctProcessing(ctSeriesID):
        try:
            ctSeriesInfo = dfCTWithRTSTRUCTS.loc[dfCTWithRTSTRUCTS["series_x"] == ctSeriesID]
            patID = ctSeriesInfo.iloc[0]["patient_ID_x"]
            print("Processing ", patID, "...")

            # Load CT from DICOM
            ctDirPath = os.path.join(imageDirPath, ctSeriesInfo.iloc[0]["folder_x"])
            try:
                ctImage = read_dicom_series(path=ctDirPath, series_id=ctSeriesID)
            except Exception as e:
                print(f"Failed to read DICOM series for {ctSeriesID}: {e}")
                return  # Skip this series and continue
        
            # Load RTSTRUCT from DICOM using regex
            segFilePath = os.path.join(imageDirPath, ctSeriesInfo.iloc[0]["file_path_y"])
            try:
                segImages = loadSegmentation(segFilePath,
                                            modality=ctSeriesInfo.iloc[0]["modality_y"],
                                            baseImageDirPath=ctDirPath,
                                            roiNames=roiNames)
                if not segImages:
                    print(f"No segmentation images found for {ctSeriesID}")
                    return  # Skip this series if no segmentation is found
            except Exception as e:
                print(f"Failed to load segmentation for {ctSeriesID}: {e}")
                return  # Skip this series and continue

            # Get the first ROI mask
            try:
                roiLabel = list(segImages.keys())[0]
                roiMask = segImages[roiLabel]
            except IndexError:
                print(f"No valid ROI masks found for {ctSeriesID}")
                return  # Skip this series if no ROI mask is found

            # Align the ROI mask to the CT image
            try:
                roiMask = alignImages(ctImage, roiMask)
            except Exception as e:
                print(f"Failed to align images for {ctSeriesID}: {e}")
                return  # Skip this series and continue

            # Only create directories and save files if segmentation exists
            patSeriesOutDir = os.path.join(outputDir, f"{patID}_{ctSeriesID}")
            if not os.path.exists(patSeriesOutDir):
                os.makedirs(patSeriesOutDir)

            ctOutDir = os.path.join(patSeriesOutDir, "CT")
            if not os.path.exists(ctOutDir):
                os.makedirs(ctOutDir)
        
            rtOutDir = os.path.join(patSeriesOutDir, "RTSTRUCT_CT")
            if not os.path.exists(rtOutDir):
                os.makedirs(rtOutDir)

            # Save CT to NIfTI
            try:
                sitk.WriteImage(ctImage, os.path.join(ctOutDir, "CT.nii.gz"))
                sitk.WriteImage(roiMask, os.path.join(rtOutDir, "GTVp.nii.gz"))
            except Exception as e:
                print(f"Failed to save NIfTI images for {ctSeriesID}: {e}")
                return  # Skip this series and continue

            # Save each negative control to NIfTI
            for type in controlTypes:
                for region in controlRegions:
                    try:
                        outFileName = f"CT_{type}_{region}.nii.gz"
                        fullOutPath = os.path.join(ctOutDir, outFileName)
                        ncCTImage = applyNegativeControl(baseImage=ctImage,
                                                         negativeControlType=type,
                                                         negativeControlRegion=region,
                                                         roiMask=roiMask,
                                                         randomSeed=randomSeed)
                        sitk.WriteImage(ncCTImage, fullOutPath)
                    except Exception as e:
                        print(f"Failed to apply negative control '{type}' in region '{region}' for {ctSeriesID}: {e}")
                        continue  # Continue with the next control if this one fails

        except Exception as e:
            print(f"Unexpected error processing {ctSeriesID}: {e}")

    Parallel(n_jobs=-1, require="sharedmem")(delayed(ctProcessing)(ctSeriesID) for ctSeriesID in ctSeriesIDList)
    print("Pipeline complete")

    return

if __name__ == "__main__":
    args = parser()
    config = yaml.safe_load(open(args.config_file))

    niftiConversion(edgeGraphPath=config['edgeGraphPath'],
                    imageDirPath=config['imageDirPath'],
                    outputDir=config['outputDir'],
                    roiNames=config['roiNameRegex'],
                    controlTypes=config['controlTypes'],
                    controlRegions=config['controlRegions'],
                    randomSeed=config['randomSeed'])
