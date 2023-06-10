

from transformers import AutoConfig
config = AutoConfig.from_pretrained('C:/Users/ELZAHBIA/Desktop/FLASK CAR PROBLEM/checkpoint-210')
num_layers = config.num_hidden_layers
print(num_layers)