from django.shortcuts import render
import numpy as np
import tensorflow 

# This is the view that you imported in the frontend/urls.py
from numpy import argmax
from keras.models import load_model
from pickle import load
from tensorflow.keras.preprocessing.image import load_img
#from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from project.settings import *

def word_for_id(integer, tokenizer):
		for word, index in tokenizer.word_index.items():
				if index == integer:
						return word
		return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
		# seed the generation process
		in_text = 'startseq'
		# iterate over the whole length of the sequence
		for i in range(max_length):
				# integer encode input sequence
				sequence = tokenizer.texts_to_sequences([in_text])[0]
				# pad input
				sequence = pad_sequences([sequence], maxlen=max_length)
				# predict next word
				yhat = model.predict([photo,sequence], verbose=0)
				# convert probability to integer
				yhat = argmax(yhat)
				# map integer to word
				word = word_for_id(yhat, tokenizer)
				# stop if we cannot map the word
				if word is None:
						break
				# append as input for generating the next word
				in_text += ' ' + word
				# stop if we predict the end of the sequence
				if word == 'endseq':
						break
		return in_text


def extract_features1(filename):
		model = InceptionV3()
		model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
		image = load_img(filename, target_size=(299, 299))
		image = img_to_array(image)
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		image = preprocess_input(image)
		feature = model.predict(image, verbose=0)
		return feature

tokenizer = load(open(os.path.join(BASE_DIR, 'content','tokenizer.pkl'), 'rb'))
max_length = 34
model = load_model(os.path.join(BASE_DIR, 'content', 'model_11.h5'))


def indexView(request):
		content = {}
		if request.method == 'POST':
				uploaded_file = request.FILES['document']
				savefile = FileSystemStorage()
				name = savefile.save(uploaded_file.name, uploaded_file)
				print(name)
				filename = os.path.join(MEDIA_ROOT, name)
				photo = extract_features1(filename)
				# generate description
				description = generate_desc(model, tokenizer, photo, max_length)
				description = description[8:-6]
				content = {'des' : description}
		else:
				messages.warning(request, 'File was not uploaded.')    
		return render(request, "frontend/index.html", content)  # notice the template used here
