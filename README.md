# Fashion-MNIST dataset training

## Information bottleneck

Information bottleneck layer is a penultimate layer in layer chain.
Training model with 10 classes, 10 samples per class, plot information bottleneck points.

<img src="information_bottleneck.gif" width="650"/>

## Confusion matrix

<img src="confusion_matrix.png" width="650"/>

## Hard-negatives (top-5 samples for ech class with maximum softmax confidence of misclass)

|    Class   |                                 top-1                                 |                                top-2                               |                                 top-3                                 |                                 top-4                                 |                                top-5                               |
|:----------:|:---------------------------------------------------------------------:|:------------------------------------------------------------------:|:---------------------------------------------------------------------:|:---------------------------------------------------------------------:|:------------------------------------------------------------------:|
| Ankle boot | ![Sneaker](./hard_neg/Ankle%20boot/hard_neg_1(Sneaker).png) (Sneaker) | ![Sandal](./hard_neg/Ankle%20boot/hard_neg_2(Sandal).png) (Sandal) | ![Sneaker](./hard_neg/Ankle%20boot/hard_neg_3(Sneaker).png) (Sneaker) | ![Sneaker](./hard_neg/Ankle%20boot/hard_neg_4(Sneaker).png) (Sneaker) | ![Sandal](./hard_neg/Ankle%20boot/hard_neg_5(Sandal).png) (Sandal) |
|     Bag    |           ![T-shirt](./hard_neg/Bag/hard_neg_1(T-shirt).png)          |           ![Dress](./hard_neg/Bag/hard_neg_2(Dress).png)           |            ![Sandal](./hard_neg/Bag/hard_neg_3(Sandal).png)           |           ![T-shirt](./hard_neg/Bag/hard_neg_4(T-shirt).png)          |         ![T-shirt](./hard_neg/Bag/hard_neg_5(T-shirt).png)         |
|    Coat    |                                                                       |                                                                    |                                                                       |                                                                       |                                                                    |
|    Dress   |                                                                       |                                                                    |                                                                       |                                                                       |                                                                    |
|  Pullover  |                                                                       |                                                                    |                                                                       |                                                                       |                                                                    |
|   Sandal   |                                                                       |                                                                    |                                                                       |                                                                       |                                                                    |
|    Shirt   |                                                                       |                                                                    |                                                                       |                                                                       |                                                                    |
|   Sneaker  |                                                                       |                                                                    |                                                                       |                                                                       |                                                                    |
|   T-shirt  |                                                                       |                                                                    |                                                                       |                                                                       |                                                                    |
|   Trouser  |                                                                       |                                                                    |                                                                       |                                                                       |                                                                    |
