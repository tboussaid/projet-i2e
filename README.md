# Image classification project

## Statoil/C-Core competition
In this competition, we were challenged to build an algorithm that automatically identifies if a remotely sensed target is a ship or iceberg. It is in fact a binary classification problem.

However, we did this competition as part of our final year engineering studies at [INSA Lyon](https://www.insa-lyon.fr/) in the energy and environment department. 

### Our best results
Given the distribution of the dataset, which is quite an equal distribution for icebergs and ships, we have mainly used the **accuracy** as the fundamental metric (*we also used **f1** and **logloss** in some cases*).

Our best result is given by a multi-input hybrid ```TensorFlow``` model which combines both numerical and image data. The numerical data is the result of different data processing phases of the image using ```Scikit-image``` and ```OpenCV```. We obtain an accuracy of nearly **94 %** on the validation dataset.

## Description of the repository
The **data** folder gathers the training dataset which was load using ```Git LFS```. We have also stored there our best CNN and MLP-CNN models in the ```.h5``` format from ```TensorFlow```. 

The ```data_visualisation``` notebook summarizes the different filters and denoising methods that we have used on the images and shows different images where we truly struggle to identify the object on them.

The ```naive_method``` notebook was our first and baseline model where we simply used different classifiers on the raw data (i.e. image matrix) without any pre-processing. We however made several **cross-validation** and **grid-search** to optimize the *hyperparameters*. The best accuracy was about **80 %** using ```SVM``` classifier.

The ```supervised_learning_final_all_features``` notebook compiles the different classifier results this time while using pre-processed data (denoised, ACP, statistic features, OpenCV features...). The best accuracy was about **87 %** using ```RandomForest``` classifier. Three other variants are also available for comparison with some slight differences in features and treatment(Rec_ratio,Pixel normalisation), thus some minuscule differences in perfomance are observed.

The ```deeplearning_explore``` implements our first ```CNN``` networks using ```Keras``` *Sequential API*. Three different architectures and transferred learning using **ResNet50** pretrained base are implemented. The best accuracy was given by a horizontal voting method (*to minimize the variance effect due to the small training dataset*) and was about **92 %**.

Finally, the ```hybrid_model``` notebook uses Keras ```functional API``` to build a more complex CNN & MLP network in which we feed the CNN branch with the images and the MLP branch with the "handcrafted" features.

The ```app.py``` assembles the different functions and classes that we used in our notebooks.

---
# Contributors
Contributions are very welcome! If you find a bug or some improvements, feel free to raise an issue and send a PR.

* **Taha BOUSSAID** - [tboussaid](https://github.com/tboussaid)
* **Tianfang LI** - [alexleeltf](https://github.com/alexleeltf)
* **Tristan PREVOST** - [Tristan-Prevost](https://github.com/Tristan-Prevost)
