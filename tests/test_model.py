import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.train import train_and_evaluate

if __name__ == "__main__":
    epochs = 20
    total_params, train_accuracy, val_accuracy = train_and_evaluate(max_epochs=epochs)

    if total_params > 20000:
        raise ValueError(f'Number of model parameters exceed 20k limit: {total_params}')

    if val_accuracy < 99.4:
        raise ValueError(f'Validation accuracy too low: {val_accuracy}%')
    else:
        print(f'Model training successful: {val_accuracy}% accuracy in {epochs} epochs')

