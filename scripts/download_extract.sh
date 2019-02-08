#!/bin/bash

# download and extract annotations/questions/glove
wget -P data/zips/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget -P data/zips/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
wget -P data/zips/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget -P data/zips/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget -P data/zips/ http://nlp.stanford.edu/data/glove.6B.zip
wget -P data/zips/ https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip

unzip data/zips/v2_Annotations_Train_mscoco.zip -d data
unzip data/zips/v2_Annotations_Val_mscoco.zip -d data
unzip data/zips/v2_Questions_Train_mscoco.zip -d data
unzip data/zips/v2_Questions_Val_mscoco.zip -d data
unzip data/zips/glove.6B.zip -d data/glove
unzip data/zips/trainval_36.zip -d data
mv data/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv data
rmdir data/trainval_36

# Now data/ should look like this:
#   data/
#     - zips/
#       - v2_XXX...zip
#       - glove...zip
#       - trainval_36.zip
#     - glove/
#       - glove...txt
#     - v2_XXX.json
#     - trainval_resnet...tsv
