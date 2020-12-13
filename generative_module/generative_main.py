import gpt2_generator
import bert_evaluator
import gpt_2_simple as gpt2
import numpy as np

"""Generative module implementation based on the following 
papers, articles and repositories:
	- (Article) Combining GPT-2 and BERT to make a fake person 
	(https://bonkerfield.org/2020/02/combining-gpt-2-and-bert/),
    - (Repository) gpt2-bert-reddit-bot 
    (https://github.com/lots-of-things/gpt2-bert-reddit-bot),
    - (Article) How to build a convincing reddit personality with 
    GPT2 and BERT 
    (https://bonkerfield.org/2020/02/reddit-bot-gpt2-bert/),
    - (Paper) Hierarchical Transformers for Long Document 
    Classification 
    (https://arxiv.org/abs/1910.10781),
    - (Article) Text Classification with BERT using Transformers 
    for long text inputs 
    (https://medium.com/analytics-vidhya/text-classification-with-bert-using-transformers-for-long-text-inputs-f54833994dfd),
    - (Article) Using BERT For Classifying Documents with Long 
    Texts 
    (https://medium.com/@armandj.olivares/using-bert-for-classifying-documents-with-long-texts-5c3e7b04573d),
    - (Repository) bert_for_long_text 
    (https://github.com/ArmandDS/bert_for_long_text)"
"""

def generate_bio(name, ocupation, threshold=0):
	"""Given a name and an occupation (both strings), generates a biography 
	and returns it and its associated minimum confidence level."""

	gen_bio = ''
	bio_score = -1

	with gpt2.start_tf_sess() as sess:
		gpt2.load_gpt2(sess, run_name='old_dates')

		# guarantee valid threshold value
		if threshold < 0:
			threshold = 0
		if threshold >= 1:
			# default to classifier's threshold
			threshold = 0.5

		prompt = name.title() + ' was a Venetian ' + ocupation.lower() + '. [SEP] '
		
		while bio_score < threshold:
			gen_bio = gpt2_generator.generate_fake_bio(prompt, sess)

			s_df = bert_evaluator.evaluate_text(gen_bio)
			realness_vals = s_df['prob_real'].values 
			bio_score = np.min(realness_vals)

	# return tuple (str generated_bio, float confidence)
	return gen_bio, bio_score


def generate_bio_full_prompt(prompt, threshold=0):
	"""Given a full prompt (a string), generates a biography and returns
	it and its associated minimum confidence level."""

	gen_bio = ''
	bio_score = -1

	with gpt2.start_tf_sess() as sess:
		gpt2.load_gpt2(sess, run_name='old_dates')

		# guarantee valid threshold value
		if threshold < 0:
			threshold = 0
		if threshold >= 1:
			# default to classifier's threshold
			threshold = 0.5
		
		while bio_score < threshold:
			gen_bio = gpt2_generator.generate_fake_bio(prompt, sess)

			s_df = bert_evaluator.evaluate_text(gen_bio)
			realness_vals = s_df['prob_real'].values 
			bio_score = np.min(realness_vals)

	# return tuple (str generated_bio, float confidence)
	return gen_bio, bio_score

def generate_random_bio(threshold=0):
	"""Generates a random biography (generating a random name and occupation) 
	and returns it and its associated minimum confidence level.."""

	gen_bio = ''
	bio_score = -1

	with gpt2.start_tf_sess() as sess:
		gpt2.load_gpt2(sess, run_name='old_dates')

		# guarantee valid threshold value
		if threshold < 0:
			threshold = 0
		if threshold >= 1:
			# default to classifier's threshold
			threshold = 0.5
		
		while bio_score < threshold:
			gen_bio = gpt2_generator.generate_random_fake_bio(sess)

			s_df = bert_evaluator.evaluate_text(gen_bio)
			realness_vals = s_df['prob_real'].values 
			bio_score = np.min(realness_vals)

	# return tuple (str generated_bio, float confidence)
	return gen_bio, bio_score

def evaluate_text(bio_text):
	"""Given a text, returns its associated confidence tuple (the mininum
	confidence level, the average confidence level, and the confidence levels
	of each chunk of the given text)."""

	s_df = bert_evaluator.evaluate_text(bio_text)
	realness_vals = s_df['prob_real'].values 

	# return tuple (float mininal_chunk_val, float avg_chunk_val, 
	#  float[] all_vals)
	return (np.min(realness_vals), np.mean(realness_vals), vals)



# def main():
# 	pass

# if __name__ == "__main__":
#     main()