echo "This assumes directory structure is: generated_datasets/not_preprocessed and preprocessed."
echo "Press any key to move items to test set..."
read dump

mkdir ./generated_datasets/preprocessed/test
for fn in tractograms TOMs beginnings_masks endings_masks seeds ends tract_masks;
do
    mkdir ./generated_datasets/preprocessed/test/$fn
done

for fn in `ls -1 ./generated_datasets/not_preprocessed/tractograms | cut -d '_' -f1,2 | sort -R | tail -5`;
do
    mv ./generated_datasets/not_preprocessed/tractograms/${fn}_CST_left.trk ./generated_datasets/preprocessed/test/tractograms/${fn}_CST_left.trk
    for subdir in TOMs beginnings_masks endings_masks seeds ends tract_masks;
    do
        mv ./generated_datasets/preprocessed/${subdir}/${fn}_CST_left.nii.gz ./generated_datasets/preprocessed/test/${subdir}/${fn}_CST_left.nii.gz
    done
done
