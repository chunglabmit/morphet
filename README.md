## MorPheT
An end-to-end framework for cell morphology phenotyping

<img width="600" alt="MorPheT" src="./data/imgs/Main_Fig_1.png">

## Installation
The anaconda environment for MorPheT can be installed using environment.yml.
```
conda env create -f environment.yml
```

Install required packages:
```
make setup.env.x
```

Other required packages (defined under .gitmodules):
```
git clone --recurse-submodules -j8 git@github.com:chunglabmit/morphet.git
```


### 

---

### GUI Tools (QT-based)
#### Launch MorPheT GUI
```shell
# entrypoint: ./src/gui/MorPheT/main.py
# to launch,
$ make launch.morphet.x
```
<img width="600" alt="MorPheT GUI" src="./data/imgs/MorPheT.png">

#### Stand-alone Data Annotator GUI
```shell
# entrypoint: ./src/gui/main.py
# to launch,
$ make launch.annotator.x set=[train/val/test]
# SET: 'train', 'val', etc
```
<img width="600" alt="annotation tool" src="./data/imgs/gui_annotator.png">

#### Model Evaluator GUI
```shell
# entrypoint: ./src/gui/evaluator/main.py
# to launch,
$ make launch.evaluator.x
```
<img width="600" alt="model evaluator" src="./data/imgs/model_evaluator.png">

#### Prediction Visualizer GUI
```shell
# entrypoint: ./src/gui/prediction/main.py
# to launch,
$ make launch.predictor.x
```
<img width="600" alt="Prediction Viewer" src="./data/imgs/prediction_viewer.png">
