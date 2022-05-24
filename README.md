# carbs-count

## Данные

- [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)

## Модели

## Результаты
| **Модель** | **MAE** |
|:----------:|:-------:|
|Baseline    |     11.49    |
| Resnet50 <br> <sub>epoch=10; linear_l=1; bs=64; depth=False </sub> |   7.646     |
|DenseNet121   <br> <sub>epoch=10; linear_l=1; bs=64; depth=False </sub>|   7.823    |
|    ResNet152   <br> <sub>epoch=10; linear_l=1; bs=64; depth=False </sub>    |     7.28    |
| ResNet152 <br> <sub>epoch=10; linear_l=1; bs=64; depth=True </sub> | 6.982 |
