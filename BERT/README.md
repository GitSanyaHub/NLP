# BERT

## BERT Linear Probing

- При обучении линейной головы хватило 4 эпохи, что дальше началось переобучение, `F1=0.93`


![train_loss.png](metrics%2Flinear-probing%2Ftrain_loss.png)
![val_loss.png](metrics%2Flinear-probing%2Fval_loss.png)
## Embedding Matrix Factorization
Факторизация эмббеддингов c параметром `E = 64`, действительно, сократила число параметров примерно на `20 млн`
- Число параметров в исходной модели: 107726601
- Число параметров в уменьшенной модели: 87363337
- Модель стала легче на: 20363264

Дообучение сошлось примерно на 5-й эпохе и показало `F1 = 0.88`

![train_loss_fuct.png](metrics%2Fembedding-factorization%2Ftrain_loss_fuct.png)
![val_loss_fuct.png](metrics%2Fembedding-factorization%2Fval_loss_fuct.png)

## Knowledge Distillation
### Обучение TinyBERT
- Сначала обучил с теми же параметрами без учителя, чтобы зафиксировать бейзлайн. Увеличил только количество эпох,
хватило `15-ти`, чтобы получить метрику `0.7`

![eval-f1-tiny-baseline.png](metrics%2Ftiny-distillation%2Feval-f1-tiny-baseline.png)
![eval-loss-tiny-baseline.png](metrics%2Ftiny-distillation%2Feval-loss-tiny-baseline.png)
![train-loss-tiny-baseline.png](metrics%2Ftiny-distillation%2Ftrain-loss-tiny-baseline.png)
- В файле `distillation` переопределил функцию подсчета ошибки согласно заданию и получил новый `DistillationTrainer`. Запустил обучение `TinyBERT` которое длилось очень много времени.



## Feed Forward Факторизация

- Нужно смотреть в сторону LoRA для фаторизации линейных слоев
