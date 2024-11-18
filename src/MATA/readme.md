# MATA*

## Build C++ extension
```
cd MATA/Astar && g++ -shared -Wl,-soname,mata -o mata.so -fPIC Mata.cpp Application.cpp
```

## AIDS:
To train MATA*:
```
python MATA/main.py --dataset AIDS --max-degree 12 --topk 4 --batch-size 128 --epochs 10000  --task 3 
```
To evaluate MATA*:
```
python MATA/main.py --dataset AIDS --max-degree 12 --topk 4 --batch-size 128 --epochs 10000  --task 3 --test
```
## Linux:
To train MATA*:
```
python MATA/main.py --dataset Linux --max-degree 12 --topk 4 --batch-size 128 --epochs 10000  --task 3 
```
To evaluate MATA*:
```
python MATA/main.py --dataset Linux --max-degree 12 --topk 4 --batch-size 128 --epochs 10000  --task 3 --test
```
## IMDB:
To train MATA*:
```
python MATA/main.py  --dataset IMDB --max-degree 40  --topk 6 --batch-size 128 --epochs 10000  --val-epochs 1000 --task 3 
```
To evaluate MATA*:
```
python MATA/main.py  --dataset IMDB --max-degree 40  --topk 6 --batch-size 128 --epochs 10000  --val-epochs 1000 --task 3 --test
```