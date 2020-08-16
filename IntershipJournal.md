
#### Individual task

Изучить различные архитектуры нейронных сетей, используемых в компьютерном зрении и выяснить, возможно ли знания, полученнные после обучения моделей, перенести на другие данные, полученные из других источников.

To study different architectures of neural networks, that were used in the computer vision, to find out if we could cross-domain generalize data on different datasets.



#### Expected result

Используя библиотеку pytorch и опираясь на научную статью, попробовать воспроизвести результаты исследования из статьи.

With the use of the PyTorch library and the knowledge gained from the article, try to reproduce the experiment from the article.

#### A brief description of achieved results

Перебирая различные архитектуры сетей для своего исследования я выбрал Resnet50. Тренируя данные на Resnet50, мне удалось получить получить схожие с результатами статьи графики, но у меня возникли сложности с тренировкой  и тестированием на Chest-14 датасете: он настолько велик и разнообразен, что результаты тренировки и тестирования были плохими, но в целом мне удалось повторить эксперимент, опубликованный в научной статье (https://arxiv.org/pdf/2002.02497v1.pdf).

Trying various architectures of the neural networks for my project, I have chosen Resnet50. Training data on Resnet50, I have achieved similar results from research, however, I had some difficulties with the training and testing on Chest-14 dataset: the result of the research did not match with the article as the dataset was too large and versatile. Nevertheless, I have re-conducted the experiment from the article (https://arxiv.org/pdf/2002.02497v1.pdf).

1. Start date: 08.06.2020 - 14.06.2020
    - Task  
    
    Познакомиться с бибилиотекам pytorch и torchvision, используя любой датасет с мультиклассовой классификацией попробовать потренировать модельку и подсчитать метрики - точность, f1, auc.    

    To study PyTorch and Torchvision; using any multi-class dataset, try to train model and calculate metrics: accuracy, f1, AUC. 

    - Result  
    
    Используя документацию к pytorch, torchvision мне удалось сделать свою нейронную, базирующуюся на VGG сеть, на датасете CIFAR100. К сожалению, результаты были неудовлетворительными, так как VGG была слишком мелкой, поэтому мне пришлось использовать WideResnet, которая намного лучше справилась с задачей. Узнал, что для Resnet сетей эффективнее всего брать learning rate 1е-3 - 1е-4, а для больших значений lr сеть перестает обучаться.

    To study the documentation of the PyTorch and Torchvision, I managed to train my model on CIFAR100, based on VGG. Unfortunately, the results were unsatisfactory because of the shallow network, so I had to use WideResnet that could deal with this task in a better way. I found out that for Resnet networks it is more effective to use the learning rate in the range from 1e-3 to 1e-4 and Resnet stops training for large values of learning rate.

2. Start date: 15.06.2020 - 21.06.2020
     - Task  
     
    Познакомится с технологиями контроля версий git и командами bash linux, научится предобрабатывать данные с использованием пакета pydicom.

    Get familiar to VCS git and bash Linux commands, to learn how to pre-process data with the use of pydicom.

    - Result  
    
    Я разобрался с vcs git, основными командами bash linux для подключения к удаленному серверу университета. Используя библиотеку pydicom я сумел прочитать файлы и преобработать их для последующего использования.

    I got used to VCS git, bash Linux commands for connecting to a remote university server. With the usage of pydicom, I have read medical images and pre-processed them for later use.  
 
3. Start date: 22.06.2020 - 28.06.2020
    - Task  
    
    Познакомится с transfer learning, узнать основные методы transfer learning и попробовать их применить к поставленной задаче.

    Get familiar with the key methods of transfer learning and apply them to the multi-class dataset.

    - Result  
    
    Используя уже предученные на ImageNet модели и используя "заморозку" начальных слоев, мне удалось на 40% увеличить точность и f1 модели, по сравнении с моделью с "разблокированными" слоями. 

    Using already pre-trained on ImageNet with blocked from changing first layers models, I managed to increase accuracy and f1 score on 40%, comparing to models without "frozen" layers. 

4. Start date: 29.06.2020 - 5.06.2020
    - Task  
    
    Воспроизвести исследование, описанное в статье (https://arxiv.org/pdf/2002.02497v1.pdf), сравнить полученные результаты с описанными в статье.
    
    Reproduce the research from the article (https://arxiv.org/pdf/2002.02497v1.pdf), compare the given results to the ones that were achieved.

    - Result  
    
    Используя Resnet50, мне удалось вопроизвести  результаты исследования, но у меня возникли трудности с тренировкой датасета Chest-14: несбалансированность и сама разнородность данных помешала в тренировке и обучении сети и из-за этого среднее значение AUC оказалось ниже, чем в статье. Задание выполнено.

    Using as a model Resnet50, I have achieved the data similar to the ones that were in the result of the research. However, I had some difficulties with the training on Chest-14 dataset: unbalance and heterogeneity of data prevented from successful training models, and because of that average AUC value became lower than it was in the article. The task was completed.
    
**Analysis of my work:**

In this project, I tried to repeat the experiment, taken from the article. The results that I achieved were close to ones in the article but still were lowed that were described in the paper, in my opinion, mainly because of shallow nn architecture and wrong methods of working with Chest-14 dataset. Having a variety of diseases in this dataset (pneumothorax, edema, etc.) I should have used it for more accurate and effective training. Also, shallow Resnet50 was too shallow for this task, I should have used more deep CNN. Finally, for checking the correctness of nn outputs, we could use backpropagation input images for showing disease localization.



**Future work:**

1. To try different CNN architectures to archive better training and testing results.

2. To switch from binary classification to multi-label classification in Chest-14 dataset.

3. Backpropagate input images to display localization of the disease. 

