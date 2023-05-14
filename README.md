# Fashion-MNIST dataset training

## Information bottleneck

Information bottleneck layer is a penultimate layer in layer chain.
Training model with 10 classes, 10 samples per class, plot information bottleneck points.

We can see that points model is "developing" and points are moving away from the center.

<img src="information_bottleneck.gif" width="650"/>

## Confusion matrix

Confusion matrix diagonally predominates, but `Sneakers` value is lower than it is expected,
because they are sometimes predicted as `Ankle boots`, which is normal from human's point of view.

<img src="confusion_matrix.png" width="650"/>

## Hard-negatives (top-5 samples for each class with maximum softmax confidence of misclass)

|    Class   |                                    top-1                                    |                                top-2                                |                                    top-3                                    |                                    top-4                                    |                                top-5                               |
|:----------:|:---------------------------------------------------------------------------:|:-------------------------------------------------------------------:|:---------------------------------------------------------------------------:|:---------------------------------------------------------------------------:|:------------------------------------------------------------------:|
| Ankle boot |    ![Sneaker](./hard_neg/Ankle%20boot/hard_neg_1(Sneaker).png) (Sneaker)    |  ![Sandal](./hard_neg/Ankle%20boot/hard_neg_2(Sandal).png) (Sandal) |    ![Sneaker](./hard_neg/Ankle%20boot/hard_neg_3(Sneaker).png) (Sneaker)    |    ![Sneaker](./hard_neg/Ankle%20boot/hard_neg_4(Sneaker).png) (Sneaker)    | ![Sandal](./hard_neg/Ankle%20boot/hard_neg_5(Sandal).png) (Sandal) |
|     Bag    |         ![T-shirt](./hard_neg/Bag/hard_neg_1(T-shirt).png) (T-shirt)        |        ![Dress](./hard_neg/Bag/hard_neg_2(Dress).png) (Dress)       |          ![Sandal](./hard_neg/Bag/hard_neg_3(Sandal).png) (Sandal)          |         ![T-shirt](./hard_neg/Bag/hard_neg_4(T-shirt).png) (T-shirt)        |    ![T-shirt](./hard_neg/Bag/hard_neg_5(T-shirt).png) (T-shirt)    |
|    Coat    |       ![Pullover](./hard_neg/Coat/hard_neg_1(Pullover).png) (Pullover)      |       ![Dress](./hard_neg/Coat/hard_neg_2(Dress).png) (Dress)       |           ![Shirt](./hard_neg/Coat/hard_neg_3(Shirt).png) (Shirt)           |           ![Shirt](./hard_neg/Coat/hard_neg_4(Shirt).png) (Shirt)           |       ![Dress](./hard_neg/Coat/hard_neg_5(Dress).png) (Dress)      |
|    Dress   |        ![T-shirt](./hard_neg/Dress/hard_neg_1(T-shirt).png) (T-shirt)       |    ![T-shirt](./hard_neg/Dress/hard_neg_2(T-shirt).png) (T-shirt)   |      ![Pullover](./hard_neg/Dress/hard_neg_3(Pullover).png) (Pullover)      |      ![Pullover](./hard_neg/Dress/hard_neg_4(Pullover).png) (Pullover)      |  ![Pullover](./hard_neg/Dress/hard_neg_5(Pullover).png) (Pullover) |
|  Pullover  |           ![Coat](./hard_neg/Pullover/hard_neg_1(Coat).png) (Coat)          |       ![Coat](./hard_neg/Pullover/hard_neg_2(Coat).png) (Coat)      |           ![Coat](./hard_neg/Pullover/hard_neg_3(Coat).png) (Coat)          |      ![T-shirt](./hard_neg/Pullover/hard_neg_4(T-shirt).png) (T-shirt)      |      ![Coat](./hard_neg/Pullover/hard_neg_5(Coat).png) (Coat)      |
|   Sandal   |       ![Sneaker](./hard_neg/Sandal/hard_neg_1(Sneaker).png) (Sneaker)       |   ![Sneaker](./hard_neg/Sandal/hard_neg_2(Sneaker).png) (Sneaker)   |       ![Sneaker](./hard_neg/Sandal/hard_neg_3(Sneaker).png) (Sneaker)       |             ![Bag](./hard_neg/Sandal/hard_neg_4(Bag).png) (Bag)             |         ![Bag](./hard_neg/Sandal/hard_neg_5(Bag).png) (Bag)        |
|    Shirt   |        ![T-shirt](./hard_neg/Shirt/hard_neg_1(T-shirt).png) (T-shirt)       |    ![T-shirt](./hard_neg/Shirt/hard_neg_2(T-shirt).png) (T-shirt)   |            ![Coat](./hard_neg/Shirt/hard_neg_3(Coat).png) (Coat)            |        ![T-shirt](./hard_neg/Shirt/hard_neg_4(T-shirt).png) (T-shirt)       |        ![Coat](./hard_neg/Shirt/hard_neg_5(Coat).png) (Coat)       |
|   Sneaker  | ![Ankle boot](./hard_neg/Sneaker/hard_neg_1(Ankle%20boot).png) (Ankle boot) |    ![Sandal](./hard_neg/Sneaker/hard_neg_2(Sandal).png) (Sandal)    | ![Ankle boot](./hard_neg/Sneaker/hard_neg_3(Ankle%20boot).png) (Ankle boot) | ![Ankle boot](./hard_neg/Sneaker/hard_neg_4(Ankle%20boot).png) (Ankle boot) |        ![Bag](./hard_neg/Sneaker/hard_neg_5(Bag).png) (Bag)        |
|   T-shirt  |     ![Pullover](./hard_neg/T-shirt/hard_neg_1(Pullover).png) (Pullover)     | ![Pullover](./hard_neg/T-shirt/hard_neg_2(Pullover).png) (Pullover) |          ![Dress](./hard_neg/T-shirt/hard_neg_3(Dress).png) (Dress)         |          ![Dress](./hard_neg/T-shirt/hard_neg_4(Dress).png) (Dress)         |    ![Sandal](./hard_neg/T-shirt/hard_neg_5(Sandal).png) (Sandal)   |
|   Trouser  |          ![Dress](./hard_neg/Trouser/hard_neg_1(Dress).png) (Dress)         |      ![Dress](./hard_neg/Trouser/hard_neg_2(Dress).png) (Dress)     |          ![Dress](./hard_neg/Trouser/hard_neg_3(Dress).png) (Dress)         |          ![Dress](./hard_neg/Trouser/hard_neg_4(Dress).png) (Dress)         |     ![Dress](./hard_neg/Trouser/hard_neg_5(Dress).png) (Dress)     |

We can see that all mispredicted samples are very similar to the true ones, for example:
    * `Ankle boots` to `Sneakers`,  to `Sandals` to `Sneakers`.
    * `Pullover` to `Coat`, `Dress` to `Pullover`, `Coat` to `Pullover` and `Shirt`
    * `Trousers` to `Dress`.
    * `Bag` is predicted both as `T-shirt`, `Dress`, `Sandals`, etc, due to unusual shape in comparison with other classes.