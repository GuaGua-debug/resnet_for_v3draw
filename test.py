import numpy as np
import pandas as pd
import seaborn
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from train import V3drawDataset, ResNet_3d_binary, Transform4D, Augmentation3D, Gamma_correction
import torch
from tqdm import tqdm
import logging
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modelname = r"save/resnet_0407"
CSVfile = r"label/test.csv"
IMAGEfile = r"data"
model_path = rf'{modelname}/best_cell_classifier.pth'
log_path = rf"{modelname}/test/evaluate_result.txt"
pic_path = rf"{modelname}/test/Confusion Matrix.png"
csv_path = rf"{modelname}/test/test_result.csv"


def calculate_metrics(preds, labels):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(preds)):
        if preds[i] == 1 and labels[i] == 1: tp += 1
        if preds[i] == 0 and labels[i] == 0: tn += 1
        if preds[i] == 1 and labels[i] == 0: fp += 1
        if preds[i] == 0 and labels[i] == 1: fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {
        "confusion_matrix": [[tp, fn], [fp, tn]],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def encode_label(label):
    encoded = 'apical' if label == 1 else 'others'
    return encoded

def get_result(label, pred):
    result = None if label == pred else 'Failed'
    return result


if __name__ == "__main__":
    logg = 0
    if logg:
        logging.basicConfig(filename=log_path, filemode='w',
                            format='%(asctime)s - %(message)s', level=logging.INFO)
        logging.info("Model Path: " + model_path)
        logging.info("CSV Path: " + CSVfile)
        logging.info("Image Path: " + IMAGEfile)

    # model = ResUnet2D(in_channels=1, num_classes=2, base_filters=32)
    model = ResNet_3d_binary(in_channels=1, num_classes=2)
    model = model.to(device)

    if logg:
        logging.info(model)

    transform = torchvision.transforms.Compose([
        Gamma_correction(2.2),
        # Mip(),
        Transform4D(),
        # Augmentation3D()
    ])

    test_dataset = V3drawDataset(images_path=IMAGEfile,
                                 labels_path=CSVfile,
                                 transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=False)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    if logg:
        logging.info("Total: {:d}".format(len(test_dataset.file_list)))
        logging.info("**Start Testing: **")

    model.eval()
    all_preds = []
    all_labels = []
    start = time.time()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if logg:
        logging.info("time: {:.4f}s, ".format((time.time() - start)))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = calculate_metrics(all_preds, all_labels)

    # print(metrics)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    if logg:
        logging.info("Test Result: ")
        logging.info("Accuracy: {:.4f}".format(metrics['accuracy']))
        logging.info("Precision: {:.4f}".format(metrics['precision']))
        logging.info("Recall: {:.4f}".format(metrics['recall']))
        logging.info("F1 Score: {:.4f}".format(metrics['f1']))


        result_list = pd.DataFrame(data=None, columns=['filename', 'label', 'pred', 'type'])
        file_list = test_dataset.file_list
        label_list = [encode_label(all_labels[i]) for i in range(len(all_labels))]
        pred_list = [encode_label(all_preds[i]) for i in range(len(all_preds))]
        type_list = [get_result(all_labels[i], all_preds[i]) for i in range(len(all_preds))]
        result_list['filename'] = file_list
        result_list['label'] = label_list
        result_list['pred'] = pred_list
        result_list['type'] = type_list
        result_list.to_csv(csv_path, index=False)


        cm = metrics['confusion_matrix']
        plt.figure(figsize=(6, 4))
        seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative'],
                    yticklabels=['Positive', 'Negative'])
        plt.xlabel('Predicted Labels', color='orange')
        plt.ylabel('True Labels', color='green')
        plt.title('Confusion Matrix')
        plt.savefig(pic_path)
        # plt.show()