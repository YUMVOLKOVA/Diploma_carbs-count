# carbs-count

## Данные

- [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)

## Модели

### ResNet
![resnet](pics/resnet.jpg)
### DenseNet
![resnet](pics/densenet.jpg)
## Результаты
|                              **Модель**                              | **MAE**  |     **MSE**      | **RMSE** |  
|:--------------------------------------------------------------------:|:--------:|:----------------:|:--------:|
| baseline <br> <sub>среднее значение по тренировочному датасету</sub> |          |                  |          | 
|           resnet152 <br> <sub>(без карт глубины)    </sub>           |  7.177   |   125.3          |  10.53   |
|           resnet152 <br> <sub>(с картами глубины)   </sub>           | **6.13** |      	92.51      |  9.238   | 
|           resnet50 <br> <sub>(без карт глубины)    </sub>            |     |           |     |
|           resnet50 <br> <sub>(с картами глубины)   </sub>            | |      	     |   | 
		|          densenet121 <br> <sub>(без карт глубины)    </sub>          |   |            |    |
|          densenet121 <br> <sub>(с картами глубины)   </sub>          |  |      	      |    | 
|          densenet201 <br> <sub>(без карт глубины)    </sub>          |   |            |    |
|          densenet201 <br> <sub>(с картами глубины)   </sub>          |  |      	      |    | 
								