a=10
p=0.1
num=1
norm=3

echo "This file only works with V1.1.0 of the TractSeg dataset. Are you sure you wish to continue?"
read dump

script_loc="../../../Programs/MITK-Diffusion-Ubuntu/MitkFiberDirectionExtraction.sh"
base_dir="../../DATASETS/TRACTSEG_105_SUBJECTS"
v1_1_dir="../../DATASETS/V1.1.0_TRACTSEG_105_SUBJECTS_V1.1.0"

output_dir="$base_dir/generated_toms"
mask_dir="$base_dir/generated_tract_masks"

for subject in `ls -1 $v1_1_dir | egrep '^[0-9]{6}$'`
do
    mkdir "$output_dir/$subject"
    for tract in `ls -1 ${v1_1_dir}/${subject}/tracts | sed 's/\.trk$//g'`
    do
        v1_1_tractogram="$v1_1_dir/$subject/tracts/${tract}.trk" # location of v1.1 tractogram
        output_fn="$output_dir/$subject/$tract"
        mask="$mask_dir/$subject/${tract}.trk.nii.gz"

        sh $script_loc -i $v1_1_tractogram -o $output_fn --mask $mask --athresh $a --peakthresh $p --numdirs $num --normalization $norm --file_ending .nii.gz
    done
done

