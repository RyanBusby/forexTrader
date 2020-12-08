import os
import json
import time
import math
import matplotlib.pyplot as plt
from data_processor import DataLoader
from model import Model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.style.use('ggplot')

def plot_results(predicted_data, true_data):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def main():
    with open('my_config.json', 'r') as f:
        configs = json.load(f)
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])

    data = DataLoader('gbpusda', configs)

    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
    	seq_len = configs['data']['sequence_length'],
    	normalise = configs['data']['normalise']
    )

    # out-of memory generative training
    steps_per_epoch =\
        math.ceil(
            (
                data.len_train - configs['data']['sequence_length']
            )\
            / configs['training']['batch_size']
        )
    model.train_generator(
    	data_gen = data.generate_train_batch(
    		seq_len = configs['data']['sequence_length'],
    		batch_size = configs['training']['batch_size'],
    		normalise = configs['data']['normalise']
    	),
    	epochs = configs['training']['epochs'],
    	batch_size = configs['training']['batch_size'],
    	steps_per_epoch = steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )

    x_test, y_test = data.get_test_data(
    	seq_len = configs['data']['sequence_length'],
    	normalise = configs['data']['normalise']
    )

    if configs['plot'] == 'multi_sequence':
        predictions_multiseq = model.predict_sequences_multiple(
            x_test,
            configs['data']['sequence_length'], configs['data']['sequence_length']
        )
        plot_results_multiple(
            predictions_multiseq,
            y_test,
            configs['data']['sequence_length']
        )

    elif configs['plot'] == 'full_sequence':
        predictions_fullseq = model.predict_sequence_full(
            x_test, configs['data']['sequence_length']
        )
        plot_results(predictions_fullseq, y_test)

    elif configs['plot'] == 'point_by_point':
        predictions_pointbypoint = model.predict_point_by_point(x_test)
        plot_results(predictions_pointbypoint, y_test)

if __name__ == '__main__':
    main()
