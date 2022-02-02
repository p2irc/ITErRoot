#!/bin/bash


#---------------------------------------------------------
# This is a bash script which was used to compare the
# results of the outputs from each of the segmentation
# approaches used in the paper.  It simply calls the
# compute_metrics.py script for each of the datasets.
#---------------------------------------------------------

# Error handling
err_report() {
    echo "Error on line $1"
    dddlog "Error" "There was an error computing the metrics."
    exit
}

trap 'err_report $LINENO' ERR

# --------------------------------------------------------
# These variables will need to be updated with the correct
# paths to the datasets on your machine.
# --------------------------------------------------------

ITER_NET="/path/to/IterNet/outputs/and/results"
SEG_ROOT="/path/to/SegRoot/outputs/and/results"
ITER_ROOT="/path/to/ITErRoot/outputs/and/results"

CROPPED="Cropped_Image_Set"
FULL="Full_Image_Set"
TEST_PATCH="Testing_Patch_Set"
VALID_PATCH="Validation_Patch_Set"

CUC="Hold_Out_Cucumber"
TEST="Test"
VALID="Validation"

HP_MODEL="Hyper_Parameter_Tuning_Model"
EX_MODEL="Expanded_Patch_Training_Model"

METRICS_OUT="path/to/place/output/All_Metrics"
GT_PATH="/path/to/GT/Datasets"

# Compute metrics for IterNet Cropped Cucumber
python compute_metrics.py "$ITER_NET/$CROPPED/$CUC/Final_Predictions/" "$GT_PATH/$CROPPED/$CUC/masks/" "$METRICS_OUT/iter_net_cropped_cucumber.json"

# Compute Metrics for IterNet Cropped Test
python compute_metrics.py "$ITER_NET/$CROPPED/$TEST/Final_Predictions/" "$GT_PATH/$CROPPED/$TEST/masks/" "$METRICS_OUT/iter_net_cropped_test.json"

# Compute Metrics for IterNet Cropped Validation
python compute_metrics.py "$ITER_NET/$CROPPED/$VALID/Final_Predictions/" "$GT_PATH/$CROPPED/$VALID/masks/" "$METRICS_OUT/iter_net_cropped_valid.json"

# Compute Metrics for IterNet Full Image Cucumber
python compute_metrics.py "$ITER_NET/$FULL/$CUC/Final_Predictions/" "$GT_PATH/$FULL/$CUC/masks/" "$METRICS_OUT/iter_net_full_cucumber.json"

# Compute Metrics for IterNet Full Image Test
python compute_metrics.py "$ITER_NET/$FULL/$TEST/Final_Predictions/" "$GT_PATH/$FULL/$TEST/masks/" "$METRICS_OUT/iter_net_full_test.json"

# Compute Metrics for IterNet Full Image Validation
python compute_metrics.py "$ITER_NET/$FULL/$VALID/Final_Predictions/" "$GT_PATH/$FULL/$VALID/masks/" "$METRICS_OUT/iter_net_full_valid.json"

# Compute Metrics for IterNet Patch Testing
python compute_metrics.py "$ITER_NET/$TEST_PATCH/" "$GT_PATH/$TEST_PATCH/masks/" "$METRICS_OUT/iter_net_test_patches.json"

# Compute Metrics for IterNet Patch Validation
python compute_metrics.py "$ITER_NET/$VALID_PATCH/" "$GT_PATH/$VALID_PATCH/masks/" "$METRICS_OUT/iter_net_valid_patches.json"

# Compute Metrics for SegRoot Cropped Cucumber
python compute_metrics.py "$SEG_ROOT/$CROPPED/$CUC/Final_Predictions/" "$GT_PATH/$CROPPED/$CUC/masks/" "$METRICS_OUT/seg_root_cropped_cucumber.json"

# Compute Metrics for SegRoot Cropped Test
python compute_metrics.py "$SEG_ROOT/$CROPPED/$TEST/Final_Predictions/" "$GT_PATH/$CROPPED/$TEST/masks/" "$METRICS_OUT/seg_root_cropped_test.json"

# Compute Metrics for SegRoot Cropped Validation
python compute_metrics.py "$SEG_ROOT/$CROPPED/$VALID/Final_Predictions/" "$GT_PATH/$CROPPED/$VALID/masks/" "$METRICS_OUT/seg_root_cropped_valid.json"

# Compute Metrics for SegRoot Full Cucumber
python compute_metrics.py "$SEG_ROOT/$FULL/$CUC/Final_Predictions/" "$GT_PATH/$FULL/$CUC/masks/" "$METRICS_OUT/seg_root_full_cucumber.json"

# Compyte Metrics for SegRoot Full Test
python compute_metrics.py "$SEG_ROOT/$FULL/$TEST/Final_Predictions/" "$GT_PATH/$FULL/$TEST/masks/" "$METRICS_OUT/seg_root_full_test.json"

# Compuet Metrics for SegRoot Full Validation
python compute_metrics.py "$SEG_ROOT/$FULL/$VALID/Final_Predictions/" "$GT_PATH/$FULL/$VALID/masks/" "$METRICS_OUT/seg_root_full_valid.json"

# Compute Metrics for SegRoot Patch Test
python compute_metrics.py "$SEG_ROOT/$TEST_PATCH/" "$GT_PATH/$TEST_PATCH/masks/" "$METRICS_OUT/seg_root_test_patches.json"

# Compute Metrics for SegRoot Patch Validation
python compute_metrics.py "$SEG_ROOT/$VALID_PATCH/" "$GT_PATH/$VALID_PATCH/masks/" "$METRICS_OUT/seg_root_valid_patches.json"

# Compute Metrics for ITErRoot HP Model Full Test
python compute_metrics.py "$ITER_ROOT/$HP_MODEL/$FULL/$TEST/" "$GT_PATH/$FULL/$TEST/masks/" "$METRICS_OUT/iter_root_hp_full_test.json"

# Compute Metrics for ITErRoot HP Model Cropped Test
python compute_metrics.py "$ITER_ROOT/$HP_MODEL/$CROPPED/$TEST/" "$GT_PATH/$CROPPED/$TEST/masks/" "$METRICS_OUT/iter_root_hp_cropped_test.json"

# Compute Metrics for ITErROot HP Model Patch Test
python compute_metrics.py "$ITER_ROOT/$HP_MODEL/$TEST_PATCH/" "$GT_PATH/$TEST_PATCH/masks/" "$METRICS_OUT/iter_root_hp_patch_test.json"

# Compute Metrics for ITErRoot EX Model Cropped Cucumber
python compute_metrics.py "$ITER_ROOT/$EX_MODEL/$CROPPED/$CUC/assembled_masks/" "$GT_PATH/$CROPPED/$CUC/masks/" "$METRICS_OUT/iter_root_ex_cropped_cucumber.json"

# Compute Metrics for ITErRoot EX Model Cropped Test
python compute_metrics.py "$ITER_ROOT/$EX_MODEL/$CROPPED/$TEST/assembled_masks/" "$GT_PATH/$CROPPED/$TEST/masks/" "$METRICS_OUT/iter_root_ex_cropped_test.json"

# Compute Metrics for ITErRoot EX Model Cropped Validation
python compute_metrics.py "$ITER_ROOT/$EX_MODEL/$CROPPED/$VALID/assembled_masks/" "$GT_PATH/$CROPPED/$VALID/masks" "$METRICS_OUT/iter_root_ex_cropped_valid.json"

# Compute Metrics for ITErRoot EX MODEL Full Cucumber
python compute_metrics.py "$ITER_ROOT/$EX_MODEL/$FULL/$CUC/assembled_masks/" "$GT_PATH/$FULL/$CUC/masks/" "$METRICS_OUT/iter_root_ex_full_cucumber.json"

# Compute Metrics for ITErRoot EX MODEL Full Test
python compute_metrics.py "$ITER_ROOT/$EX_MODEL/$FULL/$TEST/assembled_masks/" "$GT_PATH/$FULL/$TEST/masks/" "$METRICS_OUT/iter_root_ex_full_test.json"

# Compute Metrics for ITErRoot EX MODEL Full Validation
python compute_metrics.py "$ITER_ROOT/$EX_MODEL/$FULL/$VALID/assembled_masks/" "$GT_PATH/$FULL/$VALID/masks/" "$METRICS_OUT/iter_root_ex_full_valid.json"

# Compute Metrics for ITErRoot EX MODEL Patch Test
python compute_metrics.py "$ITER_ROOT/$EX_MODEL/$TEST_PATCH/" "$GT_PATH/$TEST_PATCH/masks/" "$METRICS_OUT/iter_root_ex_test_patch.json"

# Compute Metrics for ITErRoot EX MODEL Patch Validation
python compute_metrics.py "$ITER_ROOT/$EX_MODEL/$VALID_PATCH/" "$GT_PATH/$VALID_PATCH/masks/" "$METRICS_OUT/iter_root_ex_valid_patch.json"

echo "Metrics Complete!"
echo "ALl metrics have been computed."
