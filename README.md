# Fashion-MNIST dataset training

## Information bottleneck

Information bottleneck layer is a penultimate layer in layer chain.
Training model with 10 classes, 10 samples per class, plot information bottleneck points.

<img src="information_bottleneck.gif" width="650"/>

## Confusion matrix

<img src="confusion_matrix.png" width="650"/>

## Hard-negatives (top-5 samples for ech class with maximum softmax confidence of misclass)

|    Class   |                            top-1                            |                            top-2                            |                           top-3                           |                           top-4                           |                           top-5                           |
|:----------:|:-----------------------------------------------------------:|:-----------------------------------------------------------:|:---------------------------------------------------------:|:---------------------------------------------------------:|:---------------------------------------------------------:|
| Ankle boot | ![Trouser](./hard_neg/Ankle%20boot/hard_neg_1(Trouser).png) | ![Trouser](./hard_neg/Ankle%20boot/hard_neg_2(Trouser).png) | ![Trouser](./hard_neg/Ankle%20boot/hard_neg_3(Shirt).png) | ![Trouser](./hard_neg/Ankle%20boot/hard_neg_4(Shirt).png) | ![Sandal](./hard_neg/Ankle%20boot/hard_neg_5(Sandal).png) |
|     Bag    |                                                             |                                                             |                                                           |                                                           |                                                           |
|    Coat    |                                                             |                                                             |                                                           |                                                           |                                                           |
|    Dress   |                                                             |                                                             |                                                           |                                                           |                                                           |
|  Pullover  |                                                             |                                                             |                                                           |                                                           |                                                           |
|   Sandal   |                                                             |                                                             |                                                           |                                                           |                                                           |
|    Shirt   |                                                             |                                                             |                                                           |                                                           |                                                           |
|   Sneaker  |                                                             |                                                             |                                                           |                                                           |                                                           |
|   T-shirt  |                                                             |                                                             |                                                           |                                                           |                                                           |
|   Trouser  |                                                             |                                                             |                                                           |                                                           |                                                           |
