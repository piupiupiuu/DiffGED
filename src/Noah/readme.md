# Noah

To train Noah on AIDS:
```
python Noah/main.py --dataset AIDS --model-epoch-start 0 --model-epoch-end 200 --model-train 1
```
To evaluate Noah on AIDS:
```
python Noah/main.py --dataset AIDS --model-epoch-start 200 --model-epoch-end 200 --model-train 0 --beamsize 100
```
To train and evaluate Noah on other datasets, replace `AIDS` of the `dataset` parameter with `Linux` or `IMDB`.