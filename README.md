# Fruit Anomaly Detection  
This project aims to develop a neural network-based classifier to detect 
## Requirements  
- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- PIL (Python Imaging Library)  

## Dataset
For this project, I utilized the Fruit and Vegetable Disease Dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten/data)
. This dataset includes a variety of images representing both healthy and rotten fruits and vegetables, making it ideal for developing machine learning models for disease detection. Specifically, I focused on the apple images from this collection to implement an anomaly detection algorithm. The dataset comprises high-quality, manually inspected images that are categorized to facilitate tasks such as image classification and deep learning. 
    
**Examples of fresh apples:**    
![vertical_flip_Screen Shot 2018-06-08 at 5 34 21 PM](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/9a9f513e-fb60-4986-8d30-9e6bbbee7409)
![vertical_flip_Screen Shot 2018-06-08 at 5 33 18 PM](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/1b5d78d2-0f61-4807-85e6-ec3dca559659)
![saltandpepper_Screen Shot 2018-06-08 at 5 27 54 PM](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/f7596566-998c-4386-b0ae-61e1da3395dd)
![rotated_by_75_Screen Shot 2018-06-08 at 5 13 18 PM](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/34f620b3-3180-4476-bb74-96c327fe5059)
![freshApple (835)](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/b8758010-5460-47e1-a9bd-b75fea82d017)
![freshApple (426)](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/f0f5cf86-f859-4a0d-b0ad-1ada8b6b4d93)
![freshApple (211)](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/067c9447-2a96-4a3b-ae48-f9a6ce4c832d)
![FreshApple (53)](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/f820ed6e-4ccb-4f03-bc70-bfb0b3143601)


**Examples of rotten apples:**    

![vertical_flip_Screen Shot 2018-06-07 at 3 02 51 PM](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/f38de27f-3bf1-4829-af30-53171175a6dd)
![rottenApple (492)](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/50b2ca2f-2add-4644-9b05-fdad60fcdd09)
![rottenApple (460) - Copy](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/4d28251a-13f1-4ca0-9648-9b7f54ab5353)
![rottenApple (26)](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/b91f22d0-c741-4291-8505-1fa9c3c0d9bb)
![rottenApple (2)](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/2c3fb0df-8e8e-4324-b476-b129f1b2938a)
![rottenApple (1)](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/b5cda774-cdd1-45ab-9b08-dc1b65bba847)
![rotated_by_45_Screen Shot 2018-06-07 at 2 58 04 PM - Copy](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/f46cee69-a3c4-408a-b839-a394b7090088)
![rotated_by_15_Screen Shot 2018-06-08 at 2 35 03 PM - Copy](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/30264653-13c8-4f78-bc9b-df413d7d0232)
![rotated_by_15_Screen Shot 2018-06-07 at 3 00 40 PM - Copy](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/3b20e4b1-82f3-41e4-aae9-296455ebdf55)
![rotated_by_15_Screen Shot 2018-06-07 at 2 15 20 PM](https://github.com/mhkt19/FruitAnomalyDetection/assets/3819181/391be645-fb69-41f6-9d7a-77d9df21ba86)


## Method:  
This project implements an anomaly detection algorithm using a convolutional neural network (CNN) to identify rotten apples. The neural network is built using the PyTorch library and utilizes the ResNet18 model pre-trained on the ImageNet dataset. The project includes several key features such as dynamic epoch determination with early stopping, dataset splitting into training and testing sets, data transformations, and calculation of various performance metrics.
### Code Explanation  
**Model Architecture:** The project uses the ResNet18 architecture, which is a well-known convolutional neural network pre-trained on the ImageNet dataset. The final layer of ResNet18 is replaced with a fully connected layer to accommodate binary classification (defective or non-defective).  
**Dynamic Epochs:** The training loop runs for a minimum of 5 epochs and a maximum of 15 epochs. However, if the model's performance on the validation set does not improve for a specified number of consecutive epochs (patience = 3), the training stops early to prevent overfitting.  
**Best Model Saving:** During training, the model with the best validation loss is saved. This ensures that the best performing model is used for evaluation.  
**Dataset Splitting:** The dataset is divided into training and testing sets with a specified ratio (70% training and 30% testing). Additionally, the training set is further split into training and validation sets (80% training and 20% validation) to monitor the model's performance during training.  
**Folder Structure:** For each run, timestamped directories are created to store the train, test, and misclassified images. This ensures that each run's results are stored separately.  
#### Performance Metrics  
##### Evaluation Metrics:  
After training, the model is evaluated on both the training and testing datasets. The following metrics are calculated:  
+ **Accuracy:** The ratio of correctly predicted instances to the total instances.  
+ **Confusion Matrix:** A table that describes the performance of the classification model by displaying the true positive, true negative, false positive, and false negative values.  
+ **Precision:** The ratio of true positive predictions to the sum of true positive and false positive predictions.  
+ **Recall:**  The ratio of true positive predictions to the sum of true positive and false negative predictions.   
+ **Duration:** The time taken to complete the training process.  
+ **Max Memory Usage:** The peak memory usage during the training process.  
##### Multiple Runs and Averaging: 
**Multiple Runs:** The entire training and evaluation process is repeated for a specified number of runs (10 by default). This helps in assessing the model's stability and consistency across different runs.  
**Averaging Statistics:** The average values of all calculated metrics across the runs are computed and stored in a separate file. This provides a comprehensive overview of the model's performance.  
## Result:
After training, evaluation metrics such as accuracy, precision, recall, and the confusion matrix are saved in metrics.txt. Since the training and testing data are selected randomly for each run, the results are not deterministic across multiple runs. However, the average statistics over 10 runs are listed below:     
Average Train Accuracy: 97.32%    
Average Test Accuracy: 95.65%
## Configuration
The project uses a JSON configuration file (config.json) to set various parameters for the anomaly detection algorithm. Below is an explanation of each parameter included in the configuration file:    
+ **base_dir:** The base directory for the project. _(Default: '.')_    
+ **dataset_dir:** The directory where the dataset is stored._(Default: 'dataset')_    
+ **train_size_ratio:** The ratio of the dataset to be used for training. The remaining data is used for testing._(Default: '.7')_      
+ **min_epochs:** The minimum number of epochs to train the model._(Default: '5')_      
+ **max_epochs:** The maximum number of epochs to train the model._(Default: '15')_      
+ **patience:** The number of epochs to wait for an improvement in validation loss before stopping early._(Default: '3')_      
+ **num_runs:** The number of times to run the training and evaluation process for averaging results._(Default: '10')_      
+ **batch_size:** The batch size used for training the model._(Default: '32')_      
+ **learning_rate:** The learning rate for the optimizer._(Default: '.001')_      
+ **transform_resize:** The dimensions to which input images will be resized._(Default: '[224, 224]')_      
+ **transform_mean:** The mean values for normalizing the images._(Default: '[0.485, 0.456, 0.406]')_      
+ **transform_std:** The standard deviation values for normalizing the images._(Default: '[0.229, 0.224, 0.225]')_      
+ **train_val_split_ratio:** The ratio of the training dataset to be used for validation._(Default: '0.8')_      
+ **dataset_percentage:** The percentage of the original dataset to be used in each run. This helps in running the code faster during early stages of implementation._(Default: '100')_     
+ **improvement_threshold:** The threshold for considering an improvement in validation loss to reset the early stopping patience counter._(Default: '.001')_      
## Contributing:
Feel free to contribute to this project by opening issues or submitting pull requests.

## License:
This project is licensed under the MIT License.
