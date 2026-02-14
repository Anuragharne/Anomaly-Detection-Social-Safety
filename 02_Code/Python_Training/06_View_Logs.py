import json
import os
import matplotlib.pyplot as plt

# Path to your saved model logs
LOG_FILE = r"03_Models\VideoMAE_Model\trainer_state.json"

def show_results():
    if not os.path.exists(LOG_FILE):
        print("CRITICAL: No log file found.")
        print(f"Checked: {LOG_FILE}")
        print("Reason: You likely closed VS Code BEFORE training finished.")
        print("Solution: You must re-run the training script.")
        return

    with open(LOG_FILE, 'r') as f:
        data = json.load(f)

    print("-" * 50)
    print(f"TRAINING REPORT")
    print("-" * 50)
    
    history = data.get('log_history', [])
    best_acc = 0.0
    best_epoch = 0
    
    # filters to separate training logs from validation logs
    epochs = []
    val_accs = []
    train_loss = []
    
    for entry in history:
        # Check for Validation Accuracy
        if 'eval_accuracy' in entry:
            acc = entry['eval_accuracy'] * 100
            ep = entry['epoch']
            val_accs.append(acc)
            epochs.append(ep)
            print(f"Epoch {ep:.2f} | Accuracy: {acc:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                best_epoch = ep
                
        # Check for Training Loss
        if 'loss' in entry:
            train_loss.append(entry['loss'])

    print("-" * 50)
    print(f"★ BEST RESULT: {best_acc:.2f}% at Epoch {best_epoch}")
    print("-" * 50)

if __name__ == "__main__":
    show_results()