# Estimation of continuous valence and arousal levels from faces in naturalistic conditions, Nature Machine Intelligence 2021

Official implementation of the paper _"Estimation of continuous valence and arousal levels from faces in naturalistic conditions"_, Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos and Maja Pantic, published in Nature Machine Intelligence, January 2021 [[1]](#Citation).
Work done in collaboration between Samsung AI Center Cambridge and Imperial College London.

Please find a full-text, view only, version of the paper [here](https://rdcu.be/cdnWi).

The full article is available on the [Nature Machine Intelligence website](https://www.nature.com/articles/s42256-020-00280-0).

[Demo] Discrete Emotion + Continuous Valence and Arousal levels      |  [Demo] Displaying Facial Landmarks
:-------------------------------------------------------------------:|:--------------------------------------:
<img src='images/emotion_only.gif' title='Emotion' style='max-width:600px'></img>  |  <img src='images/emotion_with_landmarks.gif' title='Emotion with landmarks' style='max-width:600px'></img>


## Youtube Video

<p align="center">
  <a href="https://www.youtube.com/watch?v=J8Skph65ghM">Automatic emotion analysis from faces in-the-wild
  <br>
  <img src="https://img.youtube.com/vi/J8Skph65ghM/0.jpg"></a>
</p>


## Testing the pretrained models

The code requires the following Python packages : 

```
  Pytorch (tested on version 1.2.0)
  OpenCV (tested on version 4.1.0
  skimage (tested on version 0.15.0)
```

We provide two pretrained models : one on 5 emotional classes and one on 8 classes. In addition to categorical emotions, both models also predict valence and arousal values as well as facial landmarks.

To evaluate the pretrained models on the cleaned AffectNet test set, you need to first download the [AffectNet dataset](http://mohammadmahoor.com/affectnet/). Then simply run : 

```
  python test.py --nclass 8
```

where nclass defines which model you would like to test (5 or 8).

Please note that the provided pickle file contain the list of images (filenames) that we used for testing but not the image files.

The program will output the following results :

#### Results on AffectNet cleaned test set for 5 classes


```
 Expression
  ACC=0.82

 Valence
  CCC=0.90, PCC=0.90, RMSE=0.24, SAGR=0.85
 Arousal
  CCC=0.80, PCC=0.80, RMSE=0.24, SAGR=0.79
```

#### Results on AffectNet cleaned test set for 8 classes

```
  Expression
    ACC=0.75

  Valence
    CCC=0.82, PCC=0.82, RMSE=0.29, SAGR=0.84
  Arousal
    CCC=0.75, PCC=0.75, RMSE=0.27, SAGR=0.80
```

#### Class number to expression name

The mapping from class number to expression is as follows.

```
For 8 emotions :

0 - Neutral
1 - Happy
2 - Sad
3 - Surprise
4 - Fear
5 - Disgust
6 - Anger
7 - Contempt
```

```
For 5 emotions :

0 - Neutral
1 - Happy
2 - Sad
3 - Surprise
4 - Fear
```

## Citation

If you use this code, please cite:

```
@article{toisoul2021estimation,
  author  = {Antoine Toisoul and Jean Kossaifi and Adrian Bulat and Georgios Tzimiropoulos and Maja Pantic},
  title   = {Estimation of continuous valence and arousal levels from faces in naturalistic conditions},
  journal = {Nature Machine Intelligence},
  year    = {2021},
  url     = {https://www.nature.com/articles/s42256-020-00280-0}
}
```

[1] _"Estimation of continuous valence and arousal levels from faces in naturalistic conditions"_, Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos and Maja Pantic, published in Nature Machine Intelligence, January 2021 

## License

Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0 International Licence (CC BY-NC-ND) license.
