import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import pandas as pd

from utils import rle_encode
from Data_loader import CustomDataset
from torch.utils.data import DataLoader
from model.Unet import UNet

import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    test_dataset = CustomDataset(csv_file=os.path.join(args.data_root,'test.csv'), data_root = args.data_root, transform=transform, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    model = UNet().to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.result_root, 'model.pth'))['model_state_dict'])
    
    
    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(args.device)
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1).cpu()
            outputs = torch.argmax(outputs, dim=1).numpy()
            # batch에 존재하는 각 이미지에 대해서 반복
            for pred in outputs:
                pred = pred.astype(np.uint8)
                pred = Image.fromarray(pred) # 이미지로 변환
                pred = pred.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
                pred = np.array(pred) # 다시 수치로 변환
                # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
                
                for class_id in range(12):
                    class_mask = (pred == class_id).astype(np.uint8)
                    if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
                        mask_rle = rle_encode(class_mask)
                        result.append(mask_rle)
                    else: # 마스크가 존재하지 않는 경우 -1
                        result.append(-1)
                        
        else:
            submit = pd.read_csv(os.path.join(args.data_root, 'sample_submission.csv'))
            submit['mask_rle'] = result
            submit.to_csv(os.path.join(args.result_root, 'baseline_submit.csv'), index=False)