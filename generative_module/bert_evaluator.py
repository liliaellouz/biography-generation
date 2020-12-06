# %tensorflow_version 1.x
import tensorflow as tf
import tensorflow_hub as hub
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

import re
import csv
import sys
csv.field_size_limit(100000000)

# define a few basic variables related to the BERT model's architecture
MAX_SEQ_LENGTH = 200
DROPOUT_KEEP_PROB = 0.5
BERT_MODEL_DIR = 'checkpoint/bert'

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(
        	signature="tokenization_info", 
        	as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run(
            	[tokenization_info["vocab_file"],
            	 tokenization_info["do_lower_case"]])
      
    return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)



# define the possible label (0=False, 1=Real), and the
# columns of dataframes used to generate the inputs to
# BERT
label_list = [0, 1]
DATA_COLUMN = 'text'
LABEL_COLUMN = 'label'

# create the tokenizer
tokenizer = create_tokenizer_from_hub_module()

# define several training-related variables
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 2.0
# warm-up = time where the learning rate gradually increases
WARMUP_PROPORTION = 0.1
SAVE_CHECKPOINTS_STEPS = 300
SAVE_SUMMARY_STEPS = 100

# initiated to 1 just as a placeholder, value will be updated
# when training to reflect the appropriate amount for the existing
# training data
num_train_steps = 1
num_warmup_steps = 1

def clean_txt(text):
	"""Remove non-alphanumeric characters from the text."""
	text = re.sub("'", "",text)
	text=re.sub("(\\W)+"," ",text)
	return text

def get_split(text):
	"""Splits a text into several chunks of at most 200 words."""
	l_total = []
	l_parcial = text.split()[:200]
	l_total.append(" ".join(l_parcial))

	# while unchunkified text remains, chunkify it
	w = 1
	n = len(text.split())
	while (200 + (w-1)*150) < n:
			l_parcial = text.split()[ w*150 : (w*150 + 200) ]
			l_total.append(" ".join(l_parcial))
			w += 1
	return l_total

def csv2InputExamples(real_data_path='data/bios_old_dates.csv', 
					fake_data_path='data/fakes.csv'):
	"""Loads 2 csv files, one with real biographies data, and
	another with generated fake biographies (using '|' as the
	separator character), preprocesses the entries, and converts
	them into InputExamples (which will later be used by the
	BERT model for training)."""

	# load real biographies
	df = pd.read_csv(real_data_path)
	df.columns = ['bio']
	df['bio'].iloc[0]
	df['label'] = 1
	df['text'] = df['bio'].apply(clean_txt)

	# load fake biographies
	fakes_df = pd.read_csv(fake_data_path, sep='|')
	fakes_df.columns = ['bio']
	fakes_df['bio'].iloc[0]
	fakes_df['text'] = fakes_df['bio'].apply(clean_txt)
	fakes_df['label'] = 0

	# create a single dataframe with both real and fake bios,
	# split it into test and val
	df = pd.concat([df, fakes_df])[['text', 'label']].dropna()
	train, val = train_test_split(df, test_size=0.2, random_state=35)
	train.reset_index(drop=True, inplace=True)
	val.reset_index(drop=True, inplace=True)

	print("Training Set Shape :", train.shape)
	print("Validation Set Shape :", val.shape)
	print('***** Model output directory: {} *****'.format(BERT_MODEL_DIR))

	# split entries into chunks to prevent excessively long passages
	# of text (would greatly increase BERT model complexity otherwise)
	train['text_split'] = train[DATA_COLUMN].apply(get_split)
	val['text_split'] = val[DATA_COLUMN].apply(get_split)

	# generate train and validation dataframes with the correct format
	# to later generate InputExamples
	train_l = []
	label_l = []
	index_l =[]
	for idx,row in train.iterrows():
		for l in row['text_split']:
			train_l.append(l)
			label_l.append(row['label'])
			index_l.append(idx)
	train_df = pd.DataFrame({ DATA_COLUMN: train_l, 
								LABEL_COLUMN: label_l })
	
	val_l = []
	val_label_l = []
	val_index_l = []
	for idx,row in val.iterrows():
			for l in row['text_split']:
					val_l.append(l)
					val_label_l.append(row['label'])
					val_index_l.append(idx)
	val_df = pd.DataFrame({ DATA_COLUMN: val_l, 
							LABEL_COLUMN: val_label_l })

	# create the InputExamples
	train_InputExamples = train_df.apply(
		lambda x: bert.run_classifier.InputExample(guid=None,
						text_a = x[DATA_COLUMN], 
						text_b = None, 
						label = x[LABEL_COLUMN]), axis = 1)

	val_InputExamples = val_df.apply(
		lambda x: bert.run_classifier.InputExample(guid=None, 
						text_a = x[DATA_COLUMN], 
						text_b = None, 
						label = x[LABEL_COLUMN]), axis = 1)

	return train_InputExamples, val_InputExamples

def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
																 num_labels):
		"""Creates the BERT classification model."""

		bert_module = hub.Module(
						BERT_MODEL_HUB,
						trainable=True)
		bert_inputs = dict(
						input_ids=input_ids,
						input_mask=input_mask,
						segment_ids=segment_ids)
		bert_outputs = bert_module(
						inputs=bert_inputs,
						signature="tokens",
						as_dict=True)

		# "pooled_output", used for classification tasks on an entire
		# sequence/sentence
		output_layer = bert_outputs["pooled_output"]

		hidden_size = output_layer.shape[-1].value

		# create output layer
		output_weights = tf.get_variable(
						"output_weights", [num_labels, hidden_size],
						initializer=tf.truncated_normal_initializer(stddev=0.02))
		output_bias = tf.get_variable(
						"output_bias", [num_labels], initializer=tf.zeros_initializer())

		with tf.variable_scope("loss"):

				# use dropout to help prevent overfitting
				output_layer = tf.nn.dropout(output_layer, 
					keep_prob=DROPOUT_KEEP_PROB)

				# go from logit output to predicted labels
				logits = tf.matmul(output_layer, output_weights, 
					transpose_b=True)
				logits = tf.nn.bias_add(logits, output_bias)
				log_probs = tf.nn.log_softmax(logits, axis=-1)
				predicted_labels = tf.squeeze(tf.argmax(log_probs, 
				 	axis=-1, output_type=tf.int32))

				# convert labels into one-hot encoding
				one_hot_labels = tf.one_hot(labels, depth=num_labels, 
					dtype=tf.float32)

				# if predicting, return predicted labels and the probabiltiies
				if is_predicting:
						return (predicted_labels, log_probs)

				# if train/evaling, compute loss between predicted and actual
				# label
				per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, 
					axis=-1)
				loss = tf.reduce_mean(per_example_loss)
				return (loss, predicted_labels, log_probs)


# model_fn_builder creates the model function using the
# passed parameters
def model_fn_builder(num_labels, learning_rate, 
	num_train_steps, num_warmup_steps):
		"""Returns `model_fn` closure for TPUEstimator."""
		def model_fn(features, labels, mode, params):
				"""The `model_fn` for TPUEstimator."""

				input_ids = features["input_ids"]
				input_mask = features["input_mask"]
				segment_ids = features["segment_ids"]
				label_ids = features["label_ids"]

				is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
				
				# TRAIN and EVAL
				if not is_predicting:

						(loss, predicted_labels, log_probs) = create_model(
								is_predicting, input_ids, input_mask, segment_ids, 
								label_ids, num_labels)

						train_op = bert.optimization.create_optimizer(
										loss, learning_rate, num_train_steps, 
										num_warmup_steps, use_tpu=False)

						# Calculate evaluation metrics. 
						def metric_fn(label_ids, predicted_labels):
							"""Calculates several metrics to evaluate the 
							classifier's performance."""

							accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
							f1_score = tf.contrib.metrics.f1_score(
											label_ids,
											predicted_labels)
							auc = tf.metrics.auc(
											label_ids,
											predicted_labels)
							recall = tf.metrics.recall(
											label_ids,
											predicted_labels)
							precision = tf.metrics.precision(
											label_ids,
											predicted_labels) 
							true_pos = tf.metrics.true_positives(
											label_ids,
											predicted_labels)
							true_neg = tf.metrics.true_negatives(
											label_ids,
											predicted_labels)		 
							false_pos = tf.metrics.false_positives(
											label_ids,
											predicted_labels)		
							false_neg = tf.metrics.false_negatives(
											label_ids,
											predicted_labels)
							return {
											"eval_accuracy": accuracy,
											"f1_score": f1_score,
											"auc": auc,
											"precision": precision,
											"recall": recall,
											"true_positives": true_pos,
											"true_negatives": true_neg,
											"false_positives": false_pos,
											"false_negatives": false_neg
							}

						eval_metrics = metric_fn(label_ids, predicted_labels)

						if mode == tf.estimator.ModeKeys.TRAIN:
								return tf.estimator.EstimatorSpec(mode=mode,
										loss=loss,
										train_op=train_op)
						else:
								return tf.estimator.EstimatorSpec(mode=mode,
										loss=loss,
										eval_metric_ops=eval_metrics)
				else:
						(predicted_labels, log_probs) = create_model(
								is_predicting, input_ids, input_mask, segment_ids, 
								label_ids, num_labels)

						predictions = {
										'probabilities': log_probs,
										'labels': predicted_labels
						}
						return tf.estimator.EstimatorSpec(mode, 
							predictions=predictions)

		# Return the actual model function in the closure
		return model_fn

def finetune_model(real_data_path='data/bios_old_dates.csv', 
					fake_data_path='data/fakes.csv'):
	"""Finetunes the BERT model, using the training variable values
	that were defined at the top of the file."""

	# get training data as InputExamples
	train_InputExamples, val_InputExamples = csv2InputExamples(
		real_data_path, fake_data_path)

	# convert the InputExamples into InputFeatures (that BERT understands)
	train_features = bert.run_classifier.convert_examples_to_features(
			train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

	val_features = bert.run_classifier.convert_examples_to_features(
			val_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

	# compute train and warmup steps from batch size
	num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
	num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

	# config run, specifying output directory and number of
	# checkpoint steps to save
	run_config = tf.estimator.RunConfig(
			model_dir=BERT_MODEL_DIR,
			save_summary_steps=SAVE_SUMMARY_STEPS,
			save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

	# initialize model and estimator
	model_fn = model_fn_builder(
		num_labels=len(label_list),
		learning_rate=LEARNING_RATE,
		num_train_steps=num_train_steps,
		num_warmup_steps=num_warmup_steps)

	estimator = tf.estimator.Estimator(
		model_fn=model_fn,
		config=run_config,
		params={"batch_size": BATCH_SIZE})

	# create an input function for training
	# (drop_remainder = True -> to use TPUs)
	train_input_fn = bert.run_classifier.input_fn_builder(
			features=train_features,
			seq_length=MAX_SEQ_LENGTH,
			is_training=True,
			drop_remainder=False)

	val_input_fn = run_classifier.input_fn_builder(
			features=val_features,
			seq_length=MAX_SEQ_LENGTH,
			is_training=False,
			drop_remainder=False)

	# train the model
	print(f'Beginning Training!')
	current_time = datetime.now()
	estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
	print("Training took time ", datetime.now() - current_time)

	# evaluate the model with the validation set
	estimator.evaluate(input_fn=val_input_fn, steps=None)


def get_prediction(in_sentences, BERT_MODEL_DIR=BERT_MODEL_DIR):
	"""Returns a dataframe with each chunk and its associated
	predicted label and confidence level in the prediction."""

	# load model and estimator
	model_fn = model_fn_builder(
		num_labels=len(label_list),
		learning_rate=LEARNING_RATE,
		num_train_steps=num_train_steps,
		num_warmup_steps=num_warmup_steps)

	estimator = tf.estimator.Estimator(
		model_fn=model_fn,
		model_dir=BERT_MODEL_DIR,
		params={"batch_size": BATCH_SIZE})

	# preproccess each chunk for the BERT classifier
	labels = ["Fake", "Real"]
	input_examples = [run_classifier.InputExample(guid="", text_a = x[0], text_b = None , label = 0) 
	                for x in in_sentences] # here, "" is just a dummy label
	input_features = run_classifier.convert_examples_to_features(
		input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
	predict_input_fn = run_classifier.input_fn_builder(
	  features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)

	# run classifier on each chunk
	try:
	  predictions = estimator.predict(predict_input_fn)

	  # return predictions as a dataframe
	  return pd.DataFrame([(sentence, np.exp(prediction['probabilities'][1]), 
                        labels[prediction['labels']]) 
                        for sentence, prediction in zip(in_sentences, predictions)], 
                      columns=['bio', 'prob_real','label'])
	except:
	  predictions = estimator.predict(predict_input_fn, yield_single_examples=False)

  # return predictions as a dataframe
	return pd.DataFrame([(sentence, np.exp(prediction['probabilities'][0][1]), 
                      labels[prediction['labels']]) 
                      for sentence, prediction in zip(in_sentences, predictions)], 
                    columns=['bio', 'prob_real','label'])

def evaluate_text(text):
	"""Given a text, returns a dataframe with each chunk and 
	their associated predicted labels and confidence level 
	in each prediction."""

	# chunkify text
	chunks = get_split(clean_txt(text.replace('\n', ' ')))

	single_chunk = False
	if len(chunks) == 1:
		single_chunk = True
		# the estimator expects more than 1 chunk
		chunks += ['temporary second block'] 

	# get predictions for each result
	predictions_proposed = get_prediction(chunks)

	if single_chunk:
		return predictions_proposed[:-1].sort_values(by='prob_real')
	
	return predictions_proposed.sort_values(by='prob_real')