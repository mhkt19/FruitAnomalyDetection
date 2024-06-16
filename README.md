# Fruit Anomaly Detection  
This project aims to develop a neural network-based classifier to detect 
## Requirements  
- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- PIL (Python Imaging Library)  

## Dataset


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
