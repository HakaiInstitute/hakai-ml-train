#!/bin/bash

for in_file in \
"K_omoks/KM_K02/Oct15144_Orthomosaic_export_FriApr03205127.491370.tif" \
"K_omoks/KM_K04/kmkk04_Orthomosaic_export_FriApr03204859.358337.tif" \
"K_omoks/KM_K05/kmkk05_Orthomosaic_export_FriApr03204949.158979.tif" \
"K_omoks/KM_K08/kmkk08_Orthomosaic_export_FriApr03205021.620678.tif" \
"K_omoks/KM_K09/kmkk09_Orthomosaic_export_FriApr03204947.732226.tif" \
"Mamalilikulla/ML_K01/mlkk01beforeharvest_Orthomosaic_export_FriApr03213505.822262.tif" \
"Mamalilikulla/ML_K01/mlkk01afterharvest_Orthomosaic_export_FriApr03202513.250002.tif" \
"Mamalilikulla/Ml_K02_ML_K03_ML_K04/477f3662e3_2B0EF0D4ABOPENPIPELINE_Orthomosaic_export_FriApr03213323.485115.tif" \
"Mamalilikulla/Ml_K02_ML_K03_ML_K04/b4a248661c_2B0EF0D4ABOPENPIPELINE_Orthomosaic_export_FriApr03213354.263579.tif" \
"Mamalilikulla/Ml_K02_ML_K03_ML_K04/mlkk03_Orthomosaic_export_FriApr03202110.961192.tif" \
"Mamalilikulla/ML_K06/MLK06_Orthomosaic_export_FriApr03203112.765832.tif" \
"Mamalilikulla/ML_K08/Blowhole_Orthomosaic_export_FriApr03213634.882620.tif" \
"Mamalilikulla/ML_K08/MLK08_Orthomosaic_export_FriApr03203208.443124.tif" \
"Mamalilikulla/ML_K09/mlkk09_Orthomosaic_export_FriApr03215228.623847.tif" \
"Mamalilikulla/ML_K09/mlkk09_Orthomosaic_export_FriApr03203042.184264.tif" \
"Mamalilikulla/ML_K10/Oct2288089118_Orthomosaic_export_FriApr03203426.896932.tif" \
"Mamalilikulla/ML_K13/Oct2262395_Orthomosaic_export_FriApr03203512.867481.tif" \
"Mamalilikulla/ML_K15/mlkk15_Orthomosaic_export_FriApr03203239.299884.tif" \
"Mamalilikulla/ML_K18/mlkk17manualflight_Orthomosaic_export_FriApr03202616.154268.tif" \
"Mamalilikulla/ML_K19/MLK19_Orthomosaic_export_FriApr03203002.691924.tif" \
"Mamalilikulla/ML_K19/mlkk14_Orthomosaic_export_FriApr03213551.123555.tif" \
"Tlowitsis/TL_K01_120m/Sept27P1257_Orthomosaic_export_FriApr24170556.023288.tif" \
"Tlowitsis/TL_K01_60m/Sept27P1125_Orthomosaic_export_FriApr03192436.053821.tif" \
"Tlowitsis/TL_K02_120m/Photos1103_Orthomosaic_export_FriApr03192254.831054.tif" \
"Tlowitsis/TL_K02_60m/PhotosAug16678832_Orthomosaic_export_FriApr03192312.136860.tif" \
"Tlowitsis/TL_K03/tlwko3_Orthomosaic_export_FriApr03190120.909647.tif" \
"Tlowitsis/TL_K04/ProjectName_Orthomosaic_export_FriApr03193047.763597.tif" \
"Tlowitsis/TL_K05_60m/PhotosAug8219445_Orthomosaic_export_FriApr03191610.743912.tif" \
"Tlowitsis/TL_K05_120m/PhotosAug86218_Orthomosaic_export_FriApr03191633.011826.tif" \
"Tlowitsis/TL_K06/ChathamChannel_Orthomosaic_export_FriApr03190433.035433.tif" \
"Tlowitsis/TL_K07/PhotosAug16272353_Orthomosaic_export_FriApr03191222.738662.tif" \
"Tlowitsis/TL_K08/Photos354676_Orthomosaic_export_FriApr03190855.352574.tif" \
"We_Wai_Kum/CR_K01/Photos349444_Orthomosaic_export_FriApr03204452.085302.tif" \
"We_Wai_Kum/CR_K02/PhotosAug6161201_Orthomosaic_export_FriApr03204420.229968.tif" \
"We_Wai_Kum/CR_K03/Photos46106_Orthomosaic_export_FriApr03221531.069992.tif" \
"We_Wai_Kum/CR_K03/wwkk03_Orthomosaic_export_FriApr03204108.027258.tif" \
"We_Wai_Kum/CR_K04/Photos108160_Orthomosaic_export_FriApr03204350.150213.tif" \
"We_Wai_Kum/CR_K07/Aug30P69112_Orthomosaic_export_FriApr03204743.617145.tif" \
"We_Wai_Kum/CR_K08/Aug30P113145_Orthomosaic_export_FriApr03204019.427214.tif" \
"We_Wai_Kum/CR_K08/Aug30113145Aug19254348_Orthomosaic_export_FriApr03221444.537625.tif" \
"We_Wai_Kum/CR_K08/Photos254348_Orthomosaic_export_FriApr03221429.752196.tif" \
"We_Wai_Kum/CR_K09/Photos69125_Orthomosaic_export_FriApr03204625.507573.tif" \
"We_Wai_Kum/CR_K10/Photos3068_Orthomosaic_export_FriApr03204703.187576.tif" \
"We_Wai_Kum/CR_K11/Photos126253_Orthomosaic_export_FriApr03204558.238560.tif" \
"We_Wai_Kum/CR_K12/Photos129_Orthomosaic_export_FriApr03204734.821436.tif" \
"We_Wai_Kum/CR_K13/Aug30P168_Orthomosaic_export_FriApr03204226.636397.tif"
do
  # Create dir named after KEY
  mkdir -p "./nanwakolas/${in_file%/*}"

  in_nas_path="/mnt/H/Internal/RS/UAV/Files/Nanwakolas/2019/DroneDeployOrthos/${in_file}"
  in_local_path="./nanwakolas/${in_file}"
  out_local_path="${in_local_path%.*}_kelp.tif"
  out_nas_path="${in_nas_path%.*}_kelp.tif"
  weights_path="./kelp/train_output/weights/deeplabv3_kelp_200506.pt"

  # COPY image to KEY dir
  echo "Copying img from nas"
  cp -uv "$in_nas_path" "$in_local_path"

  # Classify img
  echo "Classifying $(basename ${in_local_path})"
  bash ./segment_kelp.sh "$in_local_path" "$out_local_path" "$weights_path"

  # Copy classification back to H drive
  # TODO: Permissions problem when doing this
  #  echo "Copying kelp output to $out_nas_path"
  #  cp -u "$out_local_path" "$out_nas_path"

  # Remove data from local dir
  echo "Cleaning up img data"
  rm $in_local_path
done