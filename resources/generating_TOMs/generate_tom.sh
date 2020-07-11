a=10
p=0.1
num=1
norm=3

echo "Are you sure you want to generate TOMs? Press any key to continue..."
read dump

script_loc="../../../../Programs/MITK-Diffusion-Ubuntu/MitkFiberDirectionExtraction.sh"
for fn in `ls -1 ../../data/PRE_SAMPLED/tractograms/*`
do
    base_fn=$(echo $fn | cut -d '/' -f6 | cut -d '.' -f1)
    v1_1_tractogram=$fn
    output_fn="../../data/PRE_SAMPLED/TOMs/$base_fn"
    mask="../../data/PRE_SAMPLED/tract_masks/${base_fn}.nii.gz"
    sh $script_loc -i $v1_1_tractogram -o $output_fn --mask $mask --athresh $a --peakthresh $p --numdirs $num --normalization $norm --file_ending .nii.gz
done

