# A small script for cloning XXX.nii.gz files into XXX_0.nii.gz, XXX_1.nii.gz, etc.
for fn in `ls -1 ./not_preprocessed/tractograms`;
do
    id=`echo $fn | cut -d '_' -f1`
    num=`echo $fn | cut -d '_' -f2`

    for subdir in TOMs beginnings_masks endings_masks seeds tract_masks;
    do
        cp ./preprocessed/${subdir}/${id}.nii.gz ./preprocessed/${subdir}/${id}_${num}.nii.gz
    done
done

for fn in `ls -1 ./not_preprocessed/tractograms`;
do
    id=`echo $fn | cut -d '_' -f1`
    for subdir in TOMs beginnings_masks endings_masks seeds tract_masks;
    do
        rm ./preprocessed/${subdir}/${id}.nii.gz
    done
done
