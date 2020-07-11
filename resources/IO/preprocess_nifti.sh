for base_fn in `ls -1 ../../data/PRE_SAMPLED/TOMs | sed 's/_DIRECTIONS.nii.gz//g'`;
do
    tom_in="../../data/PRE_SAMPLED/TOMs/${base_fn}_DIRECTIONS.nii.gz"
    tom_out="../../data/PRE_SAMPLED/preprocessed/TOMs/${base_fn}.nii.gz"
    mask_in="../../data/PRE_SAMPLED/tract_masks/${base_fn}.nii.gz"
    mask_out="../../data/PRE_SAMPLED/preprocessed/tract_masks/${base_fn}.nii.gz"
    beginning_in="../../data/PRE_SAMPLED/endings_masks/${base_fn}_beginnings.nii.gz"
    beginning_out="../../data/PRE_SAMPLED/preprocessed/beginnings_masks/${base_fn}.nii.gz"
    ending_in="../../data/PRE_SAMPLED/endings_masks/${base_fn}_endings.nii.gz"
    ending_out="../../data/PRE_SAMPLED/preprocessed/endings_masks/${base_fn}.nii.gz"

    echo "$base_fn"
    python3 preprocess.py $tom_in $tom_out $mask_in $mask_out $beginning_in $beginning_out $ending_in $ending_out
done
