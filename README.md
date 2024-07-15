# KalmanNet_SensorFusion
Practical Implementation of KalmanNet for Accurate Data Fusion in Integrated Navigation

## Flow Chart
![Flowchart](figs/Song1.png)
## 2012-11-16 Result on Map
![2012-11-16](figs/Song2.png)

## Environment Setup
```
pip install -r requirements.txt
```

## Data Preprocessing
1. We have provided data in the `./data/NCLT/processed/` that has been preprocessed (consistent with the paper).
2. If you want to re-download the data and process it, use the following steps
```
# download sensor data and ground truth
python ./data/NCLT/down.py --sen 
python ./data/NCLT/down.py --gt 
ls ./data/NCLT/download/sensor_data/*.tar.gz | xargs -n1 tar xzvf
python ./data/NCLT/preprocess.py
```
## Train and Predict
1. Train.
```
# KalmanNet
python train.py --cfg ./configs/nclt/fusion/wheel_gpsfusion_origin.py
# Split KalmanNet
python train.py --cfg ./configs/nclt/fusion/wheel_gpsfusion_split.py
```
2. Inference.  
    Reference `nclt_predict.ipynb`

## Plotting on Map

We recommend using [QGIS](https://qgis.org/en/site/) for visualization on maps.
If you only want to see our results: 
    Using QGIS to open `./QGIS/20121116.qgz`.
Else if you want to plot your results:
1. Using the code provided at the bottom of `nclt_predict.ipynb`, the coordinates are converted and saved into kml (Code is modified from https://github.com/AbhinavA10/mte546-project).
2. Open QGIS and load vector(`xxx.kml`)

## Thanks

We are very grateful to the author of https://github.com/AbhinavA10/mte546-project which has inspired us a lot!
 
