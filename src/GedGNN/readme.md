# GEDGNN

To train GEDGNN on AIDS:
```
python GedGNN/main.py --dataset AIDS --model-epoch-start 0 --model-epoch-end 200 --model-train 1
```
To evaluate GEDGNN on AIDS:
```
python GedGNN/main.py --dataset AIDS --model-epoch-start 200 --model-epoch-end 200 --model-train 0 --test-k 100
```
To train and evaluate GEDGNN on other datasets, replace `AIDS` of the `dataset` parameter with `Linux` or `IMDB`.