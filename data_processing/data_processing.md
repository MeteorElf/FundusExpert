# Data Processing

## Inference and Evaluate Using Segmentation Model

Please refer to [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) to build the segmentation environment and run the code.

You can refer to `infer_nnunet.py` to call the nnU-Net model for inference. The best iteration(test on [Messidor](https://opendatalab.org.cn/OpenDataLab/MESSIDOR)) of the segmentation model in Fundus-Engine is open sourced to [huggingface](https://huggingface.co/MeteorElf/FundusExpert_Seg) for reference.

You can refer to `eval_nnunet.py` to evaluate. 
Usage:
```
python eval_nnunet.py \
    --folder_ref gt/folder \
    --folder_pred predicted/folder \
    --output_dir output/dir
```

## Convert Segmented Pixels to Bbox Coordinates

`pixel2bbox.py` processes binary segmentation masks to generate bounding boxes for objects of interest.

Usage:
```
python pixel2bbox.py \
    --mask_folder /path/to/masks \
    --image_folder /path/to/images \
    --output_folder /path/to/output \
    --label_name "ObjectOfInterest" \
    --pixel_intensity 1 \
    --eps 160 \
    --min_samples 10 \
    --area_threshold 100
```
The mask corresponding to the image should have the same file name. 

Label_name is the name of the feature corresponding to the mask.

Pixel_intensity, eps, min_samples and area_threshold are clustering parameters.

