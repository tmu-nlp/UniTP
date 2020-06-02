from experiments.t_lstm_parse import PennOperator, train_type

train_type = train_type.copy()
train_type['sentiment_loss_weight'] = train_type['loss_weight']['orient']