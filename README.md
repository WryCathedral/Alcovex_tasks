# Alcovex_tasks

# Task-1 
In this task I've employed natural language processing, language modeling, and deep learning techniques to predict the next possible word based on the last word of a given sentence. The process involves data analysis, data pre-processing, tokenization, and building a deep learning model using LSTM's.

## Approach

### Preprocessing the data

To begin the pre-processing of the Metamorphosis dataset, we will eliminate any irrelevant data by removing the starting and ending lines. The starting line will be "One morning, when Gregor Samsa woke from troubled dreams, he found", and the ending line will be "first to get up and stretch out her young body." After this step, we will save the cleaned dataset as Metamorphosis_clean.txt and access it using utf-8 encoding. The next step in our cleaning process will involve removing any extra new lines, carriage returns, and Unicode characters that are unnecessary. Finally, to improve the training of our model and avoid confusion caused by word repetitions, we will ensure that each word is unique and occurs only once. We will accomplish this by removing any additional repetitions. 

### Tokenization 

it involves dividing large text data, essays, or corpora into smaller units such as documents, lines of text, or a collection of words. The Keras Tokenizer is a tool that enables us to transform a text corpus into a vectorized format. This can be achieved by representing each text as either a sequence of integers, where each integer corresponds to the index of a token in a dictionary, or as a vector where the coefficient for each token may be binary, based on word count, or based on tf-idf.

Next, we will convert the text data into numerical sequences to facilitate analysis. We will then construct the training dataset, with 'X' containing the input text data, and 'y' containing the output word predictions for each 'X'. The vocabulary size will be determined by extracting the length from tokenizer.word_index and adding 1 to account for padding, since 0 is a reserved value. Subsequently, we will convert our predictions in 'y' to categorical data of size equal to the vocabulary. This function is designed to convert a class vector (integers) into a binary class matrix, which will be useful in conjunction with our loss function, categorical_crossentropy.

## Creating a model

We will utilize a sequential model, which will include an embedding layer with specified input and output dimensions. Since we are predicting the next word based on only one word, we must specify an input length of 1. We will also add an LSTM layer to our architecture with 1000 units and a true return sequence to enable passing through another LSTM layer. For the subsequent LSTM layer, we will also use 1000 units, but we do not need to specify a return sequence since it defaults to false. Next, we will add a hidden layer with 1000 node units using the dense layer function with a relu activation. Lastly, we will include an output layer with a softmax activation and a specified vocabulary size, ensuring that we receive probabilities for outputs equal to the vocabulary size.

## Compile and fit

In the final step, we compile and fit our model, which involves training the model and saving the best weights as nextword1.h5. This way, we can avoid the need to repeatedly train the model and use the saved model when necessary. Although I have trained only on the training data, you may opt to train with both train and validation data. Our choice of loss function is categorical_crossentropy, which calculates the cross-entropy loss between the labels and predictions. We will use the Adam optimizer with a learning rate of 0.001 and compile our model based on the loss metric.

## Conclusion

The developed model for next word prediction on the metamorphosis dataset has achieved high-quality results, with a significant reduction in loss over the course of approximately 150 epochs. Although there is room for improvement with certain pre-processing steps and model adjustments, overall the model provides accurate predictions and performs well on the given dataset.

# Task-2

To create a Deep Learning model for image classification, we can follow these steps:

## Data preprocessing: Preprocess the given dataset to prepare it for training the model. This may include steps like resizing images to a uniform size, normalization, data augmentation, and splitting the data into training, validation, and test sets. I've separated the images into two folder which indicates the labelled dataset based on their name ending with 0 or 1

## Choose a Deep Learning Neural Network: We can choose any Deep Learning neural network for this task, like Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), or their combinations like CNN-RNN models. CNNs are usually the go-to choice for image classification tasks because they have shown exceptional performance in various image classification tasks.

## Training: Train the model using the preprocessed dataset. This involves choosing a suitable loss function, an optimizer, and a learning rate. During the training process, we feed the images into the model and adjust its weights based on the output of the loss function.

## Model evaluation: After training, we evaluate the model's performance on the validation dataset to check its accuracy and generalization. Based on the results, we can tweak the model parameters or choose a different neural network architecture.

## Testing: Finally, we test the model on the unseen test dataset to check its performance in real-world scenarios.

For this specific task, we need to build a binary classifier to distinguish between real and fake images. We can train the model on the provided dataset, which contains both real and photoshopped images. We can use any suitable neural network architecture for this task, like a simple CNN or a more complex architecture like ResNet. The goal is to achieve an accuracy of over 80% on the test dataset.


