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
This project implements an anomaly detection algorithm for identifying defects in metal surfaces using a convolutional neural network (CNN). The neural network is built using the PyTorch library and utilizes the ResNet18 model pre-trained on the ImageNet dataset. The project includes several key features such as dynamic epoch determination with early stopping, dataset splitting into training and testing sets, data transformations, and calculation of various performance metrics.
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

## Contributing:
Feel free to contribute to this project by opening issues or submitting pull requests.

## License:
This project is licensed under the MIT License.
