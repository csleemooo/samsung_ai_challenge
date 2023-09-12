import os
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from Data_loader import CustomDataset
from model.Unet import UNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default='D:\\samsung_segmentation\\training_result', type=str)
    parser.add_argument("--data_root", default='D:\\samsung_segmentation\\open', type=str)
    parser.add_argument("--device", default='cuda:0', type=str)

    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    
    transform = A.Compose([A.Resize(224, 224),
                           A.Normalize(),
                           ToTensorV2()])

    # data loader for training
    dataset = CustomDataset(csv_file=os.path.join(args.data_root, 'train_source.csv'), data_root = args.data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # data loader for validation
    val_dataset = CustomDataset(csv_file=os.path.join(args.data_root, 'val_source.csv'), data_root = args.data_root, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    
    # model 초기화
    model = UNet().to(args.device)

    # loss function과 optimizer 정의
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for epoch in range(20):  # 20 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):
            
            images = images.float().to(args.device)
            masks = masks.long().to(args.device)

            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks.squeeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        else:
            print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')
            
            with torch.no_grad():
                model.eval()
                result = []
                for images, masks in tqdm(val_dataloader):
                    images = images.float().to(args.device)
                    masks = masks.cpu().detach().numpy()
                    
                    outputs = model(images)
                    outputs = torch.softmax(outputs, dim=1).cpu()
                    outputs = torch.argmax(outputs, dim=1).detach().numpy()

                    result.append(np.mean(masks==outputs)*1)
                    
                else:
                    print(f'Epoch {epoch+1}, Evaluation accuracy: {np.mean(result)}')
    else:
        save_data = {'model_state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'args': args}

        torch.save(save_data, os.path.join(args.result_root, "model.pth"))