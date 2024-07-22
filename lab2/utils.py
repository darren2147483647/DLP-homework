# script for drawing figures, and more if needed
import matplotlib.pyplot as plt
# 繪製關係圖
def draw(accuracy_history,epochs,show_span,tag="Accuracy"):
    eps_history=range(0,epochs,show_span)
    plt.figure(figsize=(5, 3))
    plt.plot(eps_history, accuracy_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel(tag)
    plt.title(f"{tag} vs. Epoch")
    plt.grid(True)
    plt.show()
def train_test_plot(train_accuracy_history,test_accuracy_history,epochs,show_span,tag="Accuracy",tag2=""):
    eps_history=range(0,epochs,show_span)
    plt.figure(figsize=(5, 3))
    plt.plot(eps_history, train_accuracy_history, marker='o', label=f"Train {tag}")
    plt.plot(eps_history, test_accuracy_history, marker='o', label=f"Test {tag}")
    plt.xlabel("Epochs")
    plt.ylabel(tag)
    plt.title(f"Training and Validation {tag} over Epochs|{tag2}")
    plt.legend()
    plt.grid(True)
    plt.show()