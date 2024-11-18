# GENN_A*

## Build Cython extenstion
```
cd GENN_A && python3 setup.py build_ext --inplace
```

## AIDS

To pre-train GENN_A*:
``` 
python GENN_A/main.py --dataset AIDS --epochs 50000 --weight-decay 5.0e-5 --batch-size 128 --learning-rate 0.001
```
To fine-tune GENN_A*:
```
python GENN_A/main.py --enable-astar --dataset AIDS --epochs 1000 --weight-decay 5.0e-5 --batch-size 1 --learning-rate 0.001
```
To evaluate GENN_A*:
```
python GENN_A/main.py --test --dataset AIDS --enable-astar --astar-use-net --batch-size 1
```

## Linux
To pre-train GENN_A*:
``` 
python GENN_A/main.py --dataset Linux --epochs 50000 --weight-decay 5.0e-5 --batch-size 128 --learning-rate 0.001
```
To fine-tune GENN_A*:
```
python GENN_A/main.py --enable-astar --dataset Linux --epochs 1000 --weight-decay 5.0e-5 --batch-size 1 --learning-rate 0.001
```
To evaluate GENN_A*:
```
python GENN_A/main.py --test --dataset Linux --enable-astar --astar-use-net --batch-size 1
```

## IMDB
To pre-train GENN_A*:
```
python GENN_A/main.py --dataset IMDB --epochs 50000 --weight-decay 5.0e-5 --batch-size 16 --learning-rate 0.001
```
To fine-tune GENN_A*:
```
python GENN_A/main.py --enable-astar --dataset IMDB --epochs 1000 --weight-decay 5.0e-5 --batch-size 1 --learning-rate 0.001
```
To evaluate GENN_A*:
```
python GENN_A/main.py --test --dataset IMDB --enable-astar --astar-use-net --batch-size 1
```