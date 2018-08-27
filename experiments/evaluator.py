from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn import svm, datasets
import itertools
import ipdb


class Evaluator():
    def __init__(self):
        return

    def save_confussion_matrix(self, y_test, y_pred, class_names, normalize=True,
                               file_path='./saved/results/confussion_matrix.png',
                               cmap=plt.cm.Blues, title=None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        cm = confusion_matrix(y_test, y_pred)

        plt.figure()
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis])
        plt.imshow(cm, interpolation='nearest', cmap=cmap)

        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        #plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        if title:
            plt.title(title)
        plt.savefig(file_path)
        plt.close()
        return

    def save_precision(self, training_precision, validation_precision,
                       file_path='./saved/precision/precision.png',
                       title=None):
        """
        This function plots together the precision curves for training and validation steps.
        """
        plt.figure()

        x = [step[0] for step in training_precision]
        y = [step[1] for step in training_precision]
        plt.plot(x, y)
        x = [step[0] for step in validation_precision]
        y = [step[1] for step in validation_precision]
        plt.plot(x, y)
        plt.axis([0, 2000, 0, 1])

        #plt.tight_layout()
        plt.xlabel('Number of steps')
        plt.ylabel('Precision')
        if title:
            plt.title(title)
        plt.legend(['Precision @ Training', 'Precision @ Validation'], loc='lower right')
        plt.savefig(file_path)
        plt.close()

    def save_loss(self, loss_values, file_path='./saved/precision/loss.png',
                  title=None):
        """
        This function plots together the loss curves for training and validation steps.
        """
        plt.figure()

        x = [step[0] for step in loss_values]
        y = [step[1] for step in loss_values]
        plt.plot(x, y)
        plt.axis([0, 2000, 0, max(y)])

        #plt.tight_layout()
        plt.xlabel('Number of steps')
        plt.ylabel('Loss value')
        if title:
            plt.title(title)
        plt.legend(['Loss @ Training'], loc='upper right')
        plt.savefig(file_path)
        plt.close()

    def save_model_comparison(self, results, file_path='./saved/precision/precision_comparison.png'):
        models = sorted(results.keys())
        average_precision = [results[model] for model in models]
        # create barplot using matplotlib
        plt.bar(models, average_precision)
        plt.xlabel('Model names')
        plt.ylabel('Average precision')
        plt.title('Precision comparison between models')
        plt.savefig(file_path)
        plt.close()

    def print_accuracy_sunnary(self, y_test, y_pred, class_names,
                               file_path='./classification_report.txt'):
        text = classification_report(y_test, y_pred, target_names=class_names)
        with open(file_path, 'w') as file:
            print(text, file=file)
        return

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel='linear', C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    #------------------------------------------------------------------------------------------
    # Run metrics:
    evaluator = Evaluator()
    evaluator.save_confussion_matrix(y_test, y_pred,
                                     class_names, normalize=True,
                                     file_path='./confussion_matrix.png')
    evaluator.print_accuracy_sunnary(y_test, y_pred, class_names,
                                     file_path='./classification_report.txt')


