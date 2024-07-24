## MorPheT
An end-to-end framework for cell morphology phenotyping

<img alt="MorPheT" src="./data/imgs/Main_Fig_1.png">

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
<img alt="MorPheT GUI" src="./data/imgs/MorPheT.png">

#### Data Annotator within MorPheT
https://github.com/user-attachments/assets/8c4b603d-ca7a-4e78-83ad-8034857b4b49



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

https://github.com/user-attachments/assets/fbc503f1-906a-44e9-b8b5-58506c2a4830


