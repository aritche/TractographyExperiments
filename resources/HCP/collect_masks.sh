# A script to collect the nodif_brain_masks of subjects in the subjects.txt file

output_dir="../../DATASETS/TRACTSEG_105_SUBJECTS/hcp_brain_masks"
while read subject; do
    mkdir "${output_dir}/${subject}"
    aws s3 cp s3://hcp-openaccess/HCP/$subject/T1w/Diffusion/nodif_brain_mask.nii.gz ${output_dir}/${subject} --region us-east-1
done < subjects.txt
