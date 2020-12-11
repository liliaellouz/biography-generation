sys.path.append('../enhancements')
sys.path.append('../generative_module')
from Enhancements import remove_among_works, remove_repetitions, main, final_date_adjstment
import generative_main

def generate_bio(name, ocupation, threshold=0):
	"""Given a name and an occupation (both strings), generates an *enhanced* 
	biography and returns it and its associated minimum confidence level."""
	gen_bio, bio_score = generative_main.generate_bio(name, ocupation, threshold)
	return enhance_bio(gen_bio), bio_score

def generate_bio_full_prompt(prompt, threshold=0):
	"""Given a full prompt (a string), generates an *enhanced* biography and 
	returns it and its associated minimum confidence level."""
	gen_bio, bio_score = generative_main.generate_bio_full_prompt(prompt, threshold)
	return enhance_bio(gen_bio), bio_score


def generate_random_bio(threshold=0):
	"""Generates a random *enhanced* biography (generating a random name 
	and occupation) and returns it and its associated minimum confidence level."""
	gen_bio, bio_score = generative_main.generate_random_bio(threshold)
	return enhance_bio(gen_bio), bio_score

def enhance_bio(bio):
	"""Given a biography, performs the biographical enhancements on it."""
    adjusted_bio = remove_among_works(remove_repetitions(bio))
    try:
        adjusted_bio = final_date_adjstment(main(adjusted_bio))
    except:
        adjusted_bio = final_date_adjstment(adjusted_bio)

	return adjusted_bio