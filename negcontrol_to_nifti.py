# %%
from imgtools.io import read_dicom_series
from readii.loaders import loadSegmentation
from readii.image_processing import *
from readii.negative_controls import *
from readii.metadata import *
import SimpleITK as sitk
from joblib import Parallel, delayed
from typing import Optional
import os
import time
import datetime

def negControlToNIFTI(ctImage, alignedROIImage, segmentationLabel, outputDir, 
                      negControlTypeList: Optional[list] = None, crop=True, update=False, randomSeed=10):
  
    if negControlTypeList is None:
        negControlTypeList = ["shuffled_full", "randomized_full", "randomized_sampled_full",
                              "shuffled_roi", "randomized_roi", "randomized_sampled_roi",
                              "shuffled_non_roi", "randomized_non_roi", "randomized_sampled_non_roi"]
        
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    if crop:
        ctImage, alignedROIImage = getCroppedImages(ctImage, alignedROIImage, segmentationLabel)
        segNiftiOutPath = os.path.join(outputDir, "cropped_segmentation_mask.nii.gz")
    else:
        segNiftiOutPath = os.path.join(outputDir, "segmentation_mask.nii.gz")

    for controlType in negControlTypeList:
        print(controlType)
        if crop:
            outFileName = "cropped_CT_" + controlType + ".nii.gz"
        else:
            outFileName = "CT_" + controlType + ".nii.gz"
        fullOutPath = os.path.join(outputDir, outFileName)

        if os.path.exists(fullOutPath) and not update:
            print(controlType, " negative control already exists.")
        else:
            negControlImage = applyNegativeControl(nc_type = controlType,
                                                baseImage = ctImage,
                                                baseROI = alignedROIImage,
                                                roiLabel = segmentationLabel,
                                                randomSeed = randomSeed)
            sitk.WriteImage(negControlImage, fullOutPath)

    sitk.WriteImage(alignedROIImage, segNiftiOutPath)

    return


def main(imageDirPath, outputDir, imageFileListPath, segType, roiNames, negControlTypeList, crop=True, randomSeed=10):

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

        # Load CT by passing in specific series to find in a directory
        ctImage = read_dicom_series(path=ctDirPath, series_id=ctSeriesID)

        # Get list of segmentations to iterate over
        segSeriesIDList = ctSeriesInfo["series_seg"].unique()

        for segCount, segSeriesID in enumerate(segSeriesIDList):
                segSeriesInfo = ctSeriesInfo.loc[ctSeriesInfo["series_seg"] == segSeriesID]

                # Check that a single segmentation file is being processed
                if len(segSeriesInfo) > 1:
                    # Check that if there are multiple rows that it's not due to a CT with subseries (this is fine, the whole series is loaded)
                    if not segSeriesInfo.duplicated(subset=["series_CT"], keep=False).all():
                        raise RuntimeError(
                            "Some kind of duplication of segmentation and CT matches not being caught. Check seg_and_ct_dicom_list in radiogenomic_output."
                        )

                # Get absolute path to segmentation image file
                segFilePath = os.path.join(
                    imageDirPath, segSeriesInfo.iloc[0]["file_path_seg"]
                )
                # Get dictionary of ROI sitk Images for this segmentation file
                segImages = loadSegmentation(
                    segFilePath,
                    modality=segSeriesInfo.iloc[0]["modality_seg"],
                    baseImageDirPath=ctDirPath,
                    roiNames=roiNames,
                )

                # Check that this series has ROIs to extract from (dictionary isn't empty)
                if not segImages:
                    print(
                        "CT ",
                        ctSeriesID,
                        "and segmentation ",
                        segSeriesID,
                        " has no ROIs or no ROIs with the label ",
                        roiNames,
                        ". Moving to next segmentation.",
                    )

                else:
                    # Loop over each ROI contained in the segmentation to perform radiomic feature extraction
                    for roiImageName in segImages:
                        # Get sitk Image object for this ROI
                        roiImage = segImages[roiImageName]

                        # Exception catch for if the segmentation dimensions do not match that original image
                        try:
                            # Check if segmentation just has an extra axis with a size of 1 and remove it
                            if roiImage.GetDimension() > 3 and roiImage.GetSize()[3] == 1:
                                roiImage = flattenImage(roiImage)

                            # Check that image and segmentation mask have the same dimensions
                            if ctImage.GetSize() != roiImage.GetSize():
                                # Checking if number of segmentation slices is less than CT
                                if ctImage.GetSize()[2] > roiImage.GetSize()[2]:
                                    print(
                                        "Slice number mismatch between CT and segmentation for",
                                        patID,
                                        ". Padding segmentation to match.",
                                    )
                                    roiImage = padSegToMatchCT(
                                        ctDirPath, segFilePath, ctImage, roiImage
                                    )
                                else:
                                    raise RuntimeError(
                                        "CT and ROI dimensions do not match."
                                    )

                        # Catching CT and segmentation size mismatch error
                        except RuntimeError as e:
                            print(str(e))

                        alignedROIImage = alignImages(ctImage, roiImage)
                        segmentationLabel = getROIVoxelLabel(alignedROIImage)

                        completeOutputPath = os.path.join(outputDir, patID) 
                        
                        negControlToNIFTI(ctImage, alignedROIImage, segmentationLabel, completeOutputPath, negControlTypeList, crop, randomSeed=randomSeed)

    print("Pipeline complete")
        

if __name__ == "__main__":
    start = time.time()

    # SCRIPT SETUPT VARIABLES HERE
    # Directory that DICOMS are located
    imageDirPath = "/Users/katyscott/Documents/RADCURE/"
    # Where to save converted niftis
    outputDir = "/Users/katyscott/Documents/RADCURE/nc_niftis/"
    # Output from med-imagetools run csv
    imageFileListPath = "/Users/katyscott/Documents/RADCURE/.imgtools/imgtools_RADCURE.csv"
    segType = "RTSTRUCT"

    # If segmentation is RTSTRUCT, need ROI name
    roiNames = "GTVp*" 

    # Which negative controls to save out
    negControlTypeList = ["shuffled_roi", "randomized_roi", "randomized_sampled_roi"]

    # Whether to crop the image before generating negative control 
    crop = True

    # Random seed to have reproducible negative controls
    randomSeed = 10

    main(imageDirPath, outputDir, imageFileListPath, segType, roiNames, negControlTypeList, crop, randomSeed)
    print(time.time() - start)

