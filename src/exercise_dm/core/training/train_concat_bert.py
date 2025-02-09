import json
import os
import random
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import DistilBertModel, BertTokenizer
from sklearn.model_selection import train_test_split


# For reproducibility
SEED_INT = 123
os.environ['PYTHONHASHSEED'] = str(SEED_INT)
np.random.seed(SEED_INT)
random.seed(SEED_INT)
torch.manual_seed(SEED_INT)
TIMESTAMP = str(datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))

# load context code mapping
with open('data/code_context_map_v1.json', 'r') as f:
    context_code_mapping = json.load(f)

context_mode = 'text'  # 'code' or 'text'

def bert_tokenizer(input_data_path):
    df = pd.read_csv(input_data_path, sep=',', keep_default_na=False)

    print(df)
    texts_token = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    for sentence1, sentence2, context in tqdm(zip(df['main'], df['target'], df['context']), total=len(df)):
        if context_mode == 'code':
            input_elements = [sentence1, sentence2, context]
        elif context_mode == 'text':
            input_elements = [sentence1, sentence2, context_code_mapping[context]]
        tok = tokenizer(input_elements, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
        texts_token.append(tok)
    df['text_token'] = texts_token
    return df



class DistilBertConcatMultiPassageClassifier(nn.Module):

    def __init__(self, passage_count, class_count, dropout):
        super(DistilBertConcatMultiPassageClassifier, self).__init__()
        self.passage_count = passage_count
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-cased')
        self.concat_dropout = nn.Dropout(dropout)
        self.concat_linear = nn.Linear(768 * passage_count, 768 * passage_count)
        self.concat_relu = nn.ReLU()
        self.mixer_dropout = nn.Dropout(dropout)
        self.mixer_linear = nn.Linear(768 * passage_count, 768 * passage_count)
        self.mixer_relu = nn.ReLU()
        self.classif_dropout = nn.Dropout(dropout)
        self.classif_linear = nn.Linear(768 * passage_count, class_count)
        self.classif_relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, masks):
        ## if shape of input_ids is [2,512] which is 2 dimension, it is converted into [2,1,512] where 1 is the passage_count
        if input_ids.ndimension()==2:
           input_ids = torch.reshape(input_ids, [input_ids.size()[0], self.passage_count, input_ids.size()[1]])
        distilbert_output = self.distilbert(input_ids=input_ids[:,0], attention_mask=masks[:,0])
        hidden_state = distilbert_output[0]
        concat_outputs_bert = hidden_state[:, 0]
        for i in range(1, self.passage_count):
            distilbert_output = self.distilbert(input_ids=input_ids[:,i], attention_mask=masks[:,i])
            hidden_state = distilbert_output[0]
            pooler_output = hidden_state[:, 0]
            concat_outputs_bert = torch.cat((concat_outputs_bert, pooler_output), dim=1)
        concat_dropout_output = self.concat_dropout(concat_outputs_bert)
        concat_linear_output = self.concat_linear(concat_dropout_output)
        concat_relu_output = self.concat_relu(concat_linear_output)
        mixer_dropout_output = self.mixer_dropout(concat_relu_output)
        mixer_linear_output = self.mixer_linear(mixer_dropout_output)
        mixer_relu_output = self.mixer_relu(mixer_linear_output)
        classif_dropout_output = self.classif_dropout(mixer_relu_output)
        classif_linear_output = self.classif_linear(classif_dropout_output)
        #final_layer = self.classif_relu(classif_linear_output)
        #prob = self.softmax(final_layer)
        return classif_linear_output


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [np.float32(label) for label in df['score']]
        self.texts = [tok for tok in df['text_token']]      
        
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


def quatization_label(label):
    if label < 0.125:
        return 0
    elif label >= 0.125 and label < 0.375:
        return 1
    elif label >= 0.375 and label < 0.625:
        return 2
    elif label >= 0.625 and label < 0.875:
        return 3
    else:
        return 4

def train(model, train_data, val_data, learning_rate, epochs, batch_size, kpi_save_path):
    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size, num_workers=10, drop_last=True)
    device = torch.device("mps")
    model = model.to(device)

    
    # Define criterion
    criterion = nn.BCEWithLogitsLoss() 

    # Deinfine optimizer
    optimizer = Adam(model.parameters(), lr= learning_rate)

    val_accuracy_best_checkpoint = -1
    val_accuracy_values = []
    epoch_best_checkpoint = -1
    file_name_best_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt_folder")
    if os.path.isdir(file_name_best_path):
        shutil.rmtree(file_name_best_path)
    file_name_best_checkpoint = ''
    file_names_saved_checkpoints = []
    recent_model_degradations = []

    
    for epoch_num in range(epochs):
        model.train()
        print('start epoch {}, number of iteration is {}'.format(epoch_num + 1, len(train_dataloader)))
        total_loss_train = 0
        y_true_train, y_pred_train = [], []

        
        # NOTE: training
        training_iter = 0
        for train_input, train_label in tqdm(train_dataloader):
            training_iter += 1
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].to(device)
            
            output = torch.squeeze(model(input_id, mask))

            batch_loss = criterion(output, train_label.float())
            total_loss_train += batch_loss.item()
            
            # check average loss every 1000 iter
            if training_iter%1000 == 0:
                print(f'ave loss: {total_loss_train/training_iter}')
            
            for pred_output, true_score in zip(output, train_label):
                # if we use BCEWithLogitsLoss, the output is without sigmoid, we need to apply sigmoid here
                pred_score = 1/(1 + np.exp(-pred_output.to('cpu').detach().numpy()))
                y_pred_train.append(quatization_label(pred_score))
                y_true_train.append(quatization_label(true_score))
            
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        report = classification_report(
            np.array(y_true_train),
            np.array(y_pred_train),
            output_dict=False,
            target_names= ['Unrelated', 'Somewhat related', 'Synonyms', 'Close synonyms', 'Very close'])
        
        # recoding the kpi during training
        with open(kpi_save_path, 'a') as f:
            f.write(f'-------------training epoch {epoch_num + 1}-------------\n')
            f.write(f'{report}\n')

        
        # NOTE: validation
        model.eval()
        with torch.no_grad():
            y_true_val, y_pred_val = [], []
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].to(device)
                output = torch.squeeze(model(input_id, mask))
                batch_loss = criterion(output, val_label.float())
                for pred_output, true_score in zip(output,val_label):
                    pred_score = 1/(1 + np.exp(-pred_output.to('cpu').detach().numpy()))
                    y_pred_val.append(quatization_label(pred_score))
                    y_true_val.append(quatization_label(true_score))
        
            report = classification_report(
                np.array(y_true_val),
                np.array(y_pred_val),
                output_dict=False,
                target_names=['Unrelated', 'Somewhat related', 'Synonyms', 'Close synonyms', 'Very close'])
        
        # recoding the kpi during training
        with open(kpi_save_path, 'a') as f:
            f.write(f'-------------validation epoch {epoch_num + 1}-------------\n')
            f.write(f'{report}\n')


        # early termination condition 
        _, _, f1, _ = precision_recall_fscore_support(np.array(y_true_val),
                                    np.array(y_pred_val),
                                    average='micro')
        
        validation_accuracy = f1
        val_accuracy_values.append(validation_accuracy)
        epoch_stats_str = f'validation_f1: {f1}'
        
        CONVERGENCE_THRESHOLD = 0.001
        MAX_MODEL_DEGRADATIONS = 3
        MIN_NUM_EPOCHES = 4

        if validation_accuracy > val_accuracy_best_checkpoint:
            val_accuracy_best_checkpoint = validation_accuracy
            epoch_best_checkpoint = epoch_num
            if not os.path.exists(file_name_best_path):
                os.makedirs(file_name_best_path, exist_ok=True)
            file_name_best_checkpoint = os.path.join(file_name_best_path, f"checkpoint_{TIMESTAMP}_{epoch_num}.pt")
            torch.save(model, file_name_best_checkpoint)
            file_names_saved_checkpoints.append(file_name_best_checkpoint)

        if len(val_accuracy_values) >= 2:
            try:
                current_improvement = (val_accuracy_values[-1] - val_accuracy_values[-2]) / val_accuracy_values[-2]
            except Exception:
                current_improvement = 0
            if current_improvement >= 0:
                epoch_stats_str += f' (+{100*current_improvement:.2f}%)'
            else:
                epoch_stats_str += f' ({100*current_improvement:.2f}%)'
            print(epoch_stats_str)
            if current_improvement < 0:
                recent_model_degradations.append(epoch_num)
                if len(recent_model_degradations) >= 2:
                    print('Number of recent degradations currently taken into account for early stopping: {} (epochs {})'.format(
                        len(recent_model_degradations),
                        f"{', '.join(str(rd + 1) for rd in recent_model_degradations[:-1])} and {recent_model_degradations[-1] + 1}"
                    ))
            elif recent_model_degradations and epoch_num - recent_model_degradations[-1] > MAX_MODEL_DEGRADATIONS:
                # "Forget" oldest degradation on record if MAX_MODEL_DEGRADATIONS + 1 epochs pass with no further degradations
                recent_model_degradations.pop(0)

            if len(val_accuracy_values) >= MIN_NUM_EPOCHES and current_improvement < CONVERGENCE_THRESHOLD:
                try:
                    previous_improvement = (val_accuracy_values[-2] - val_accuracy_values[-3]) / val_accuracy_values[-3]
                except Exception:
                    previous_improvement = 0
                if previous_improvement < CONVERGENCE_THRESHOLD:
                    print(f'Improvements in last two epochs have been below {100 * CONVERGENCE_THRESHOLD}%. Stopping training...')
                    break
                elif len(recent_model_degradations) >= MAX_MODEL_DEGRADATIONS:
                    print(f'Improvement in last epoch was below {100 * CONVERGENCE_THRESHOLD}% and the model has recently degraded {len(recent_model_degradations)} times. Stopping training...')
                    break
        else:
            print(epoch_stats_str)
    

    # Save the best model
    if epoch_num != epoch_best_checkpoint:
        print(f'Reloading best checkpoint ({epoch_best_checkpoint+1}-th epoch)...')
        model = torch.load(file_name_best_checkpoint)
    for fn in file_names_saved_checkpoints:
        if fn!=file_name_best_checkpoint:
            os.remove(fn)


def start_training(input_data_path, kpi_save_path):

    data_df = bert_tokenizer(input_data_path)

    # load all data
    df = data_df.sample(frac=1, random_state=105)
    
    # Split data into train and test (80% train, 20% test)
    train_df, val_test_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)
    
    # create a customized concatenated bert model
    model = DistilBertConcatMultiPassageClassifier(
        passage_count=3,
        class_count=1,
        dropout=0.1)

    print(f'Training size: {len(train_df)} | Training size: {len(val_df)} | Testing size: {len(test_df)}')
    
    train(model = model, 
        train_data = train_df, 
        val_data = val_df, 
        learning_rate = 5e-5, 
        epochs = 6, 
        batch_size = 8, 
        kpi_save_path = kpi_save_path
        )
    
        
if __name__ == "__main__":
    kpi_save_path = f'model/concat_bert/concat_bert_kpi.txt'
    
    start_training(input_data_path="data/data.csv",  kpi_save_path=kpi_save_path)
        
