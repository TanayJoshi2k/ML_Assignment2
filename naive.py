from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_word_freq(data):
    freq = dict()
    for document in data:
        for word in document:
            freq[word] = freq.get(word, 0) + 1
    return freq

def display_metrics(accuracy, confusion_matrix, precision, recall):
    print(accuracy, precision, recall)
    print(confusion_matrix)
    plt.figure(figsize=(4, 3))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

class DataLoader:
    def __init__(self, train_pos_ratio, train_neg_ratio, test_pos_ratio, test_neg_ratio):
        self.alpha = 1
        self.train_pos_docs, self.train_neg_docs, self.vocab = load_training_set(train_pos_ratio, train_neg_ratio)
        self.test_pos, self.test_neg = load_test_set(test_pos_ratio, test_neg_ratio)
        self.pos_dict = get_word_freq(self.train_pos_docs)
        self.neg_dict = get_word_freq(self.train_neg_docs)

def log_model(dataloader, example, alpha):
    total = len(dataloader.train_pos_docs) + len(dataloader.train_neg_docs)

    y_pos_prob = len(dataloader.pos_dict) / total
    y_neg_prob = len(dataloader.neg_dict) / total

    n_pos = sum(dataloader.pos_dict.values()) + alpha * len(dataloader.pos_dict)
    n_neg = sum(dataloader.neg_dict.values()) + alpha * len(dataloader.neg_dict)

    log_y_pos_prob, log_y_neg_prob = np.log(y_pos_prob), np.log(y_neg_prob)

    for word in example:
        count_pos = dataloader.pos_dict.get(word, 0) 
        count_neg = dataloader.neg_dict.get(word, 0) 
        
        log_y_pos_prob += np.log((count_pos + alpha)/ n_pos)
        log_y_neg_prob += np.log((count_neg + alpha)/ n_neg)

    if log_y_pos_prob > log_y_neg_prob:
        return 1
    elif log_y_pos_prob == log_y_neg_prob:
        random_integer = np.random.randint(0, 2)
        # Map 0 to -1 and 1 to 1
        random_choice = 2 * random_integer - 1
        return random_choice
    return -1

def posterior_model(dataloader, example, alpha=0):
    total = len(dataloader.train_pos_docs) + len(dataloader.train_neg_docs)

    y_pos_prob = len(dataloader.pos_dict) / total
    y_neg_prob = len(dataloader.neg_dict) / total

    n_pos = sum(dataloader.pos_dict.values()) 
    n_neg = sum(dataloader.neg_dict.values())

    for word in set(example):
        y_pos_prob *= dataloader.pos_dict.get(word, 0)/n_pos
        y_neg_prob *= dataloader.neg_dict.get(word, 0)/n_neg

    if y_pos_prob > y_neg_prob:
        return 1
    elif y_pos_prob == y_neg_prob:
        random_integer = np.random.randint(0, 2)
        # Map 0 to -1 and 1 to 1
        random_choice = 2 * random_integer - 1
        return random_choice
    return -1

def calculate_metrics(model, dataloader, alpha):
    correct_predictions = 0
    total_examples = len(dataloader.test_pos) + len(dataloader.test_neg)

    confusion_matrix = pd.DataFrame({'Pred Positive':[0], 'Pred Negative': [0]}, 
                                index=['Real Positive', 'Real Negative'])

    for example in dataloader.test_pos:
        if model(dataloader, example, alpha) == 1:
            correct_predictions += 1
            confusion_matrix.loc['Real Positive', 'Pred Positive'] += 1
        else:
            confusion_matrix.loc['Real Positive', 'Pred Negative'] += 1

    for example in dataloader.test_neg:
        if model(dataloader, example, alpha) == -1:
            correct_predictions += 1
            confusion_matrix.loc['Real Negative', 'Pred Negative'] += 1
        else:
            confusion_matrix.loc['Real Negative', 'Pred Positive'] += 1

    accuracy = correct_predictions / total_examples

    TP = confusion_matrix['Pred Positive']['Real Positive']
    FP = confusion_matrix['Pred Positive']['Real Negative']
    FN = confusion_matrix['Pred Negative']['Real Positive']
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    return accuracy, confusion_matrix, precision, recall

# print("Question 1: Probabilities v/s Log-probabilities")

# q1_dl = DataLoader(0.2, 0.2, 0.2, 0.2)
# alpha = 0
# accuracy, confusion_matrix, precision, recall = calculate_metrics(posterior_model, q1_dl, alpha)
# display_metrics(accuracy, confusion_matrix, precision, recall)

# accuracy, confusion_matrix, precision, recall = calculate_metrics(log_model, q1_dl, alpha)
# display_metrics(accuracy, confusion_matrix, precision, recall)

# print("=======================================")

print("Question 2: alpha v/s Accuracy graph")
accuracies = {}
alpha = 0.0001
q2_dl = DataLoader(0.2, 0.2, 0.2, 0.2)
while alpha < 1001:
    acc, cm, precision, recall = calculate_metrics(log_model, q2_dl, alpha)
    print(f"alpha: {alpha}, acc: {acc}")
    accuracies[alpha] = acc
    alpha *= 10

sns.set_style("whitegrid")

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(x=accuracies.keys(), y=accuracies.values(), marker='o', color='blue')
plt.xscale('log')  # Set log scale for the x-axis
plt.xlabel('Alpha (α)')
plt.ylabel('Accuracy on Test Set')
plt.title('Model Accuracy vs. Alpha')

for i, (alpha, accuracy) in enumerate(zip(accuracies.keys(), accuracies.values())):
    plt.annotate(f'{accuracy:.3f}', (alpha, accuracy), textcoords="offset points", xytext=(0,10), ha='center')
plt.show()

print("=======================================")

# print("Question 3: 100% train and 100% test")
# q3_dl = DataLoader(1, 1, 1, 1)
# alpha = 1
# acc, cm, precision, recall = calculate_metrics(log_model, q3_dl, alpha)
# print(acc, precision, recall)
# print(cm)

# print("=======================================")

# print("Question 4: 50% train and 50% test")

# q4_dl = DataLoader(0.5, 0.5, 1, 1)
# alpha = 1
# accuracy, confusion_matrix, precision, recall = calculate_metrics(log_model, q4_dl, alpha)
# display_metrics(accuracy, confusion_matrix, precision, recall)
    
# # print("=======================================")

print("Question 6: 10% positive, 50% negative train and 100% test")
q6_dl = DataLoader(0.1, 0.5, 1, 1)
alpha = 1
accuracy, confusion_matrix, precision, recall = calculate_metrics(log_model, q6_dl, alpha)
display_metrics(accuracy, confusion_matrix, precision, recall)
