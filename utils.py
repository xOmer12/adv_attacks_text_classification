import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.manifold import TSNE
import seaborn as sns

def get_cls_token_embedding(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]
    return cls_token