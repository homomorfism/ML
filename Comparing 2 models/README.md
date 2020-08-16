
####In this project I tried to compare evaluating 2 neural networks trained and tested on different datasets

Tested models:
- Simple CNN with repeated blocks (conv, batchnorm, relu)
- Pretrained on ImageNet Resnet18


Data:
- training dataset: https://www.kaggle.com/c/Kannada-MNIST train.csv
- validation dataset: https://www.kaggle.com/c/Kannada-MNIST Dig-MNIST.csv
- test dataset: https://www.kaggle.com/c/digit-recognizer train.csv

Results:
1. Simple CNN with repeated blocks (conv, batchnorm, relu)
   
    1.1 Training results
    
        Accuracy: 0.73017578125 
        Precision: 0.7704144202543628 
        Recall: 0.73017578125 
        F1: 0.7306299214326741 
        ROC-AUC 0.961757230758667
        
    1.2 Testing results
    
        Accuracy: 0.3842857142857143 
        Precision: 0.37085412169575904 
        Recall: 0.3842857142857143 
        F1: 0.36391692181486684 
        ROC-AUC 0.7082598236918157
        
2. Pretrained on ImageNet Resnet18
    
    2.1 Training results
    
        Accuracy: 0.7080078125 
        Precision: 0.7643630948450687 
        Recall: 0.7080078125 
        F1: 0.7123425109952704 
        ROC-AUC 0.9604425430297854
    
    2.2 Testing results
    
        Accuracy: 0.2895238095238095 
        Precision: 0.33555712454470227 
        Recall: 0.2895238095238095 
        F1: 0.29571277851227407 
        ROC-AUC 0.6895053501813354