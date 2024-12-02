import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from src.models import UnifiedDeepFM
from src.data import *
from src.data.context_data import *


class DeepFMTrainer:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.params = args.model_args[args.model]
        
        # Dimensions
        self.n_user = len(data['label2idx']['user'])
        self.n_item = len(data['label2idx']['item'])
        self.n_genre = len(data['label2idx']['genre']) + 1
        self.n_writer = len(data['label2idx']['writer'])
        self.n_director = len(data['label2idx']['director'])
        self.n_year = 10

    def _prepare_dataloader(self, dataframe, batch_size, shuffle=True):
        dataset = InteractionDataset(
            dataframe=dataframe,
            n_user=self.n_user,
            n_item=self.n_item,
            n_genre=self.n_genre,
            n_writer=self.n_writer,
            n_director=self.n_director,
            n_year=self.n_year
        )
        padding_idx = self.n_user + self.n_item
        custom_collate_fn = partial(collate_fn, padding_idx=padding_idx)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)

    def train_model(self):
        if not self.args.predict:
            train_df, valid_df = self.data['train'], self.data['valid']
            print('----- Loading Train Data -----')
            train_loader = self._prepare_dataloader(train_df, self.args.dataloader.batch_size, shuffle=self.args.dataloader.shuffle)

            print('----- Loading Validation Data -----')
            valid_loader = self._prepare_dataloader(valid_df, self.args.dataloader.batch_size // 2, shuffle=False)
        else:
            train_df = self.data['total']
            print('----- Loading Total Data -----')
            train_loader = self._prepare_dataloader(train_df, self.params['batch_size'], shuffle=True)
            valid_loader = None

        print('----- Initializing Model -----')
        model = UnifiedDeepFM(
            input_dims=[self.n_user, self.n_item, self.n_genre, self.n_writer, self.n_director, self.n_year],
            embedding_dim=self.params['embed_dim'],
            mlp_dims=self.params['mlp_dims'],
            drop_rate=self.params['dropout']
        )
        model.to(self.device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        print('----- Training and Validation -----')
        for epoch in range(self.params['epochs']):
            # Training phase
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.params['epochs']} - Training"):
                user_ids, item_ids, genres, writers, directors, years, interaction = batch
                
                # Forward pass
                outputs = model(user_ids, item_ids, genres, writers, directors, years).squeeze()
                loss = criterion(outputs, interaction)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

            # Validation phase
            if valid_loader:
                model.eval()
                val_loss, val_correct, val_total = 0, 0, 0
                with torch.no_grad():
                    for batch in tqdm(valid_loader, desc="Validating"):
                        user_ids, item_ids, genres, writers, directors, years, interaction = [t.to(self.device) for t in batch]
                        
                        outputs = model(user_ids, item_ids, genres, writers, directors, years).squeeze()
                        val_loss += criterion(outputs, interaction).item()

                        # Accuracy
                        preds = (outputs > 0.5).float()
                        val_correct += (preds == interaction).sum().item()
                        val_total += interaction.size(0)

                avg_val_loss = val_loss / len(valid_loader)
                val_accuracy = val_correct / val_total
                print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        return model

    def generate_prediction_base(self):
        print('----- Generating Prediction Base -----')
        all_items = self.data['total']['item'].unique()
        metadata_df = pd.DataFrame({'item': all_items})

        # Merge metadata
        for df in [self.data['genre'], self.data['writer'], self.data['director'], self.data['year']]:
            metadata_df = pd.merge(metadata_df, df, how='left', on='item')

        metadata_df.fillna('unknown', inplace=True)
        metadata_df['year'] = metadata_df['year'].apply(lambda x: int(x) if x != 'unknown' else 'unknown')
        metadata_df['director'] = metadata_df['director'].apply(lambda x: [] if x == 'unknown' else x)
        metadata_df['writer'] = metadata_df['writer'].apply(lambda x: [] if x == 'unknown' else x)

        # Generate user-specific prediction base
        predict_base = []
        non_negative = self.data['total'][self.data['total']['interaction'] == 1]
        for user_id in tqdm(non_negative['user'].unique(), desc="Generating predictions"):
            seen_items = set(non_negative[non_negative['user'] == user_id]['item'])
            unseen_items = set(metadata_df['item']) - seen_items

            user_predict_df = pd.DataFrame({
                'user': [user_id] * len(unseen_items),
                'item': list(unseen_items),
                'interaction': [0] * len(unseen_items)
            })
            user_predict_df = pd.merge(user_predict_df, metadata_df, how='left', on='item')
            predict_base.append(user_predict_df)

        return pd.concat(predict_base, ignore_index=True)

    def evaluate(self, model, predict_base, top_k=10):
        print('----- Evaluating Model -----')
        model.eval()
        predictions = []

        with torch.no_grad():
            for user_id in tqdm(self.data['total']['user'].unique(), desc="Evaluating users"):
                predict_df = predict_base[predict_base['user'] == user_id]

                predict_loader = self._prepare_dataloader(predict_df, self.params['batch_size'], shuffle=False)
                user_predictions = []

                for batch in predict_loader:
                    user_ids, item_ids, genres, writers, directors, years, interaction = [t.to(self.device) for t in batch]
                    outputs = model(user_ids, item_ids, genres, writers, directors, years).squeeze()

                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)

                    user_predictions.extend(zip(item_ids.cpu().numpy(), outputs.cpu().numpy()))

                user_predictions = sorted(user_predictions, key=lambda x: x[1], reverse=True)[:top_k]
                predictions.extend([(user_id, item_id, score) for item_id, score in user_predictions])

        return pd.DataFrame(predictions, columns=['user', 'item', 'score'])


