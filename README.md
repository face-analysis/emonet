# Estimation of continuous valence and arousal levels from faces in naturalistic conditions, Nature Machine Intelligence 2021

Official implementation of the paper _"Estimation of continuous valence and arousal levels from faces in naturalistic conditions"_, Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos and Maja Pantic, published in Nature Machine Intelligence, January 2021 [[1]](#citing).
Work done in collaboration between Samsung AI Center Cambridge and Imperial College London.

Please find the full article on the [Nature Machine Intelligence website](https://www.nature.com/articles/s42256-020-00280-0).

[Demo] Discrete Emotion + Continuous Valence and Arousal levels      |  [Demo] Displaying Facial Landmarks
:-------------------------------------------------------------------:|:--------------------------------------:
<img src='images/emotion_only.gif' title='Emotion' style='max-width:600px'></img>  |  <img src='images/emotion_with_landmarks.gif' title='Emotion with landmarks' style='max-width:600px'></img>


## Youtube Video

<p align="center">
  <a href="https://www.youtube.com/watch?v=EqBn7oApMI4">Automatic emotion analysis from faces in-the-wild
  <br>
  <img src="https://img.youtube.com/vi/EqBn7oApMI4/0.jpg"></a>
</p>


## Testing the pretrained models

The code requires the following Python packages : 

```
  Pytorch (tested on version 1.2.0)
  OpenCV (tested on version 4.1.0
  skimage (tested on version 0.15.0)
```

We provide two pretrained models : one on 5 emotional classes and one on 8 classes. In addition to categorical emotions, both models also predict valence and arousal values as well as facial landmarks.

To evaluate the pretrained models on the cleaned test sets, simply run : 

```
  python test.py --nclass 8
```

where nclass defines which model you would like to test (5 or 8).

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
