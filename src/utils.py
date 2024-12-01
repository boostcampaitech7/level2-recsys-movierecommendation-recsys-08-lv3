import random
import os
import numpy as np
import torch

def seed_everything(args):
        '''
        [description]
        seed 값을 고정시키는 함수입니다.

        [arguments]
        seed : seed 값
        '''
        seed=args.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def save_to_csv(args, data, predictions):
    # Convert user and item indices back to original labels
    idx2user = data['idx2label']['user']
    idx2item = data['idx2label']['item']
    
    # Map predictions back to their original labels
    predictions['user'] = predictions['user'].apply(lambda x: idx2user[x])
    predictions['item'] = predictions['item'].apply(lambda x: idx2item[x])
    
    # Drop the score column
    predictions = predictions.drop(columns=['score'])
    
    # Define the output directory and file path
    output_dir = args.output_path
    output_file = os.path.join(output_dir, 'predictions.csv')
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    
    # Save predictions to CSV
    predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

