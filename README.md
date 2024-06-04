# Fingerprint-Recognition

1. Fingerprint image preprocessing
2. Extract minutiae from processed image (endpoints, bifurcation points)
3. Derive a match score by comparing the two processed images
4. Predict the train image with the highest match score for a specific test image
<br>

## Usage
- process a fingerprint image and extract features
```
import my_fingerprint as fp
endpoints, bifurcations, minutiae = fp.get_fp_feature(image, flg_show=True)
```


- match 2 images, derive the match score, and check the matching points
```
import my_fingerprint as fp
distances, match_score = fp.match_finger(features_A, features_B, 10, flg_show=True, img_A=img_test, img_B=img_train)
```
