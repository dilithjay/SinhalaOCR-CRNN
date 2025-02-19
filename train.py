import json
import os
import glob
import torch

from sklearn import model_selection

import config
import dataset
import engine
from model import ResNetRNNModel

from pprint import pprint


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = []
    for j in x:
        if fin == [] or j != fin[-1]:
            fin.append(j)
    return fin


def decode_predictions(preds):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            temp.append(k)
        tp = remove_duplicates(temp)
        cap_preds.append(tp)
    return cap_preds

def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.jpg"))
    with open(config.TARGET_JSON_PATH, 'r') as f:
        targets_dict = json.load(f)
    targets_1, targets_2, targets_3 = [], [], []
    for _, target_tuple_list in targets_dict.items():
        t_1, t_2, t_3 = zip(*target_tuple_list)
        targets_1.append(list(map(lambda x: x + 1, t_1)))
        targets_2.append(list(map(lambda x: x + 1, t_2)))
        targets_3.append(list(map(lambda x: x + 1, t_3)))
    
    (
        train_imgs,
        test_imgs,
        train_targets_1,
        test_targets_1,
        train_targets_2,
        test_targets_2,
        train_targets_3,
        test_targets_3
    ) = model_selection.train_test_split(
        image_files, targets_1, targets_2, targets_3, test_size=0.1, random_state=42
    )

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_imgs,
        targets=(train_targets_1, train_targets_2, train_targets_3),
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
        # collate_fn=collate_fn
    )
    test_dataset = dataset.ClassificationDataset(
        image_paths=test_imgs,
        targets=(test_targets_1, test_targets_2, test_targets_3),
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        # collate_fn=collate_fn
    )

    model = ResNetRNNModel()
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    
    best_loss = 10 ** 5
    
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        valid_preds, test_loss = engine.eval_fn(model, test_loader)
        valid_text_preds = [[], [], []]
        for vp in valid_preds:
            for i, pred in enumerate(vp):
                valid_text_preds[i] += decode_predictions(pred)
        print('-----------')
        combined_1 = list(zip(test_targets_1, valid_text_preds[0]))
        combined_2 = list(zip(test_targets_2, valid_text_preds[1]))
        combined_3 = list(zip(test_targets_3, valid_text_preds[2]))
        print('Lower:')
        pprint(combined_1[:5])
        print('Middle:')
        pprint(combined_2[:5])
        print('Upper:')
        pprint(combined_3[:5])
        
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model, config.MODEL_SAVE_PATH)
        
        print(
            f"Epoch={epoch}/{config.EPOCHS}, Train Loss={train_loss}, Test Loss={test_loss}"
        )
        scheduler.step(test_loss)


if __name__ == "__main__":
    run_training()
