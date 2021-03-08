# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's winter 2021 semester.

Your name: *Klara Båstedt*

*Answer all questions in the notebook here.  You should also write whatever high-level documentation you feel you need to here.*

### Part 1

I choose to count only the 3000 most frequent words in the features. (But I do this in part 3, not in part 1.)

### Part 2

My solution to avoid an index error for the last NE is not optimal since it adds 5 end-symbols and does not add the last few words to the features. Since this only affects this last NE and I couldn't think of an easy way around the problem, I decided to leave it like this.

### Part 5

It is clear that the model performs better on the training data than on the test data. The model seems to be better at recognizing the more common classes when tested on the test data compared to the less common classes.

### Bonus Part A

The weakest performing classes when the model is tested on the test data are 'art', 'eve' and 'nat'. The model does not manage to correctly predict them even once. An explaination for this could be that there are only 2, 7 and 3 instances of these classes in the testing data which makes it hard so say something general about the performance of these classes. However, these classes are infrequent in the data as a whole which should affect their performance since they are given fewer examples to train on.

### Bonus Part B

I decided to add the POS-tags of the context words to the features and count them in the same way I count the words. The features will therefore be of length 20 instead of 10.

The function a2.bonusb does not return anything, but it prints the two confusion matrices as well as an example of what the features look like with POS-tags.