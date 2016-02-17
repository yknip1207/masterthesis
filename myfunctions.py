# -*- coding: utf8 -*-
from os import listdir
import string
import jsonrpclib
from gensim.models import word2vec
import time
from sklearn.cluster import KMeans
import re
import gensim, logging, csv
import sys

# 讀檔名
def get_filename_list(folder_path):
	file_name_list = [f.split('.')[0] for f in listdir(folder_path) if '.DS_Store' not in str(f)]
	return file_name_list

# 把 [u'The', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd'...}] 的dict多加depedency 和 parent的 tag
def zip_indexeddepen_and_words(indexeddependencies, words):
	# 用單字的位置當key由小到大來sort indexeddep
	sorted_indexeddependencies = sorted(indexeddependencies, key=lambda k: int(k[-1].rsplit('-', 1)[-1]))
	# sorted_indexeddependencies = [
	# 								[u'root', u'ROOT-0', u'female-6'], 
	# 								[u'det', u'patient-2', u'The-1'], 
	# 								...
	# 								[u'conj_and', u'headache-10', u'dizziness-12']
	# 							   ]
	# words = [
	# 			[u'The', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'3', u'CharacterOffsetBegin': u'0', u'PartOfSpeech': u'DT', u'Lemma': u'the'}], 
	# 			[u'patient', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'11', u'CharacterOffsetBegin': u'4', u'PartOfSpeech': u'NN', u'Lemma': u'patient'}], 
	# 			...
	# 			[u'headache', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'63', u'CharacterOffsetBegin': u'55', u'PartOfSpeech': u'NN', u'Lemma': u'headache'}], [u'and', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'67', u'CharacterOffsetBegin': u'64', u'PartOfSpeech': u'CC', u'Lemma': u'and'}], [u'dizziness', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'77', u'CharacterOffsetBegin': u'68', u'PartOfSpeech': u'NN', u'Lemma': u'dizziness'}], 
	# 			[u'.', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'78', u'CharacterOffsetBegin': u'77', u'PartOfSpeech': u'.', u'Lemma': u'.'}]
	# 		  ]
	sorted_indexeddependencies_dict = {}
	for sortdep in sorted_indexeddependencies:
		# sortdep ： [u'root', u'ROOT-0', u'female-6']
		sorted_indexeddependencies_dict[int(sortdep[-1].rsplit('-', 1)[-1])] = sortdep[:-1]
	
	# sorted_indexeddependencies_dict = {1: [u'det', u'patient-2'], 2: [u'nsubj', u'female-6'], 3: [u'cop', u'female-6'], 4: [u'det', u'female-6'], 5: [u'amod', u'female-6'], 6: [u'root', u'ROOT-0'], 8: [u'prep_with', u'female-6'], 10: [u'prep_of', u'complaints-8'], 12: [u'conj_and', u'headache-10']}
	# print(d)

	for i in range(0, len(words)):
		if i+1 in sorted_indexeddependencies_dict:
			# print(str(i+1) +' in d')
			# print(sorted_indexeddependencies_dict[i+1][0], sorted_indexeddependencies_dict[i+1][1])
			# print(words[i][0], words[i][1])
			# words[i]: [u'The', {u'NamedEntityTag': u'O', u'Lemma': u'the'}]
			# words[i][1]: words[i]的dict裡多增加兩種key/val: Dependency, ParentWord
			words[i][1][u'Dependency'] = sorted_indexeddependencies_dict[i+1][0]
			words[i][1][u'ParentWord'] = sorted_indexeddependencies_dict[i+1][1].rsplit('-',1)[0]
			# words[i][1][u'ParentIndex'] = sorted_indexeddependencies_dict[i+1][1].rsplit('-',1)[1]
			if words[i][1][u'ParentWord'] == 'ROOT':
				words[i][1][u'ParentPOS'] = u'NO_PARENT_POS'
			else:
				parent_index = int(sorted_indexeddependencies_dict[i+1][1].rsplit('-',1)[1]) - 1
				words[i][1][u'ParentPOS'] = words[parent_index][1][u'PartOfSpeech']
		else:
			# 標點符號
			if words[i][0] in string.punctuation or words[i][0] in ['-LRB-','-RRB-','-LSB-','-RSB-','-LCB-','-RCB-']:
				words[i][1][u'Dependency'] = u'NO_DEP'
				words[i][1][u'ParentWord'] = u'NO_PARENT_WORD'
				words[i][1][u'ParentPOS'] = u'NO_PARENT_POS'
			else:
				words[i][1][u'Dependency'] = u'COLLAPSE'
				words[i][1][u'ParentWord'] = u'NO_PARENT_WORD'
				words[i][1][u'ParentPOS'] = u'NO_PARENT_POS'
	# print(words)

	return words
# loads(server.parse(line))的時候把所有空格都忽略掉了
# 要補len(trim掉的空白)跟len(prev_contect)回charoffset
# input1 : words_with_dep = [ ['The', {'POS': 'NN', 'Dep': 'nsubj'...}], ['Patient', {...}]  ]
# input2 : line = 目前讀進來的那一行
# input3 : prev_context
def shift_charoffset_in_words_with_depend(words_with_dep, line, prev_context):
	# left_trimmed_space_length = 0
	# if len(line) > 1 and not line.isspace():
	left_trimmed_space_length = len(line) - len(line.lstrip())

	# fo.write(str(int(feature_dict['CharacterOffsetBegin']) + len(prev_context) + left_trimmed_space_length) + ' ')
	# word: 'The'
	# feature_dict = {'POS': 'NN', 'Dep': 'nsubj'...}
	for i in range(0, len(words_with_dep)):
		word, feature_dict = words_with_dep[i][0], words_with_dep[i][1]
		# print('space=' + str(left_trimmed_space_length) + '     feature_dict[\'CharacterOffsetBegin\']:' + feature_dict['CharacterOffsetBegin'] + '    feature_dict[\'CharacterOffsetEnd\']:' + feature_dict['CharacterOffsetEnd'] + '   prev:' + str(len(prev_context)))
		newBegin = int(feature_dict['CharacterOffsetBegin']) + len(prev_context) + left_trimmed_space_length
		newEnd = int(feature_dict['CharacterOffsetEnd']) + len(prev_context) + left_trimmed_space_length
		words_with_dep[i][1]['CharacterOffsetBegin'] = str(newBegin)
		words_with_dep[i][1]['CharacterOffsetEnd'] = str(newEnd)
		words_with_dep[i][1]['WordLength'] = str(newEnd-newBegin)
		# words_with_dep[i][1]['CharacterOffsetBegin'] = unicode(str(newBegin), 'utf-8')
		# words_with_dep[i][1]['CharacterOffsetEnd'] = unicode(str(newEnd), 'utf-8')
		# words_with_dep[i][1]['WordLength'] = unicode(str(newEnd-newBegin), 'utf-8')
	return words_with_dep	




# 輸入一句子，把裡面所有 [**xxxxxx**] 全部代換掉
def remove_noise(sentence):
	# print('!!',str(len(sentence)), sentence)
	sbblock_index_list_list = get_square_bracket_block(sentence)
	for sbblock_index_list in sbblock_index_list_list:
		s_index, e_index, sbblock = sbblock_index_list[0], sbblock_index_list[1], sbblock_index_list[2]
		if 'NAME' in sbblock.upper():
			sentence = sentence[:s_index] + ( 'NAME'+' '*(len(sbblock)-len('NAME')) ) + sentence[e_index:]
		elif 'NUMBER' in sbblock.upper():
			sentence = sentence[:s_index] + ( 'NUMBER'+' '*(len(sbblock)-len('NUMBER')) ) + sentence[e_index:]
		elif 'LOCATION' in sbblock.upper():
			sentence = sentence[:s_index] + ( 'LOCATION'+' '*(len(sbblock)-len('LOCATION')) ) + sentence[e_index:]	
		elif 'PHONE' in sbblock.upper():
			sentence = sentence[:s_index] + ( 'PHONE'+' '*(len(sbblock)-len('PHONE')) ) + sentence[e_index:]
		elif 'HOSPITAL' in sbblock.upper():
			sentence = sentence[:s_index] + ( 'HOSPITAL'+' '*(len(sbblock)-len('HOSPITAL')) ) + sentence[e_index:]
		elif 'INFO' in sbblock.upper():
			sentence = sentence[:s_index] + ( 'INFO'+' '*(len(sbblock)-len('INFO')) ) + sentence[e_index:]
		elif re.findall( '[0-9]{4}[-][0-9]{2}[-][0-9]{2}', sbblock):
			sentence = sentence[:s_index] + ( '2016-01-27'+' '*(len(sbblock)-len('2016-01-27')) ) + sentence[e_index:]
		elif re.findall( '[0-9]{2}[-][0-9]{2}', sbblock):
			sentence = sentence[:s_index] + ( '01-27'+' '*(len(sbblock)-len('01-27')) ) + sentence[e_index:]
		elif re.findall( '[0-9]{4}', sbblock):
			sentence = sentence[:s_index] + ( '2016'+' '*(len(sbblock)-len('2016')) ) + sentence[e_index:]
	punc_list = ['[', ']', '!', '@', '#', '$' ,'%', '^', '&', '*', '(', ')', '_', '?', '<', '>', '{', '}', '~', '+', '=', '\'', '"', '/', ',', ':', ';']
	# sentence = re.sub('[!@#$%^&*()_?<>[]{}~+=\'"/,:;]', '',sentence)
	for p in punc_list:
		if p in sentence:
			sentence = sentence.replace(p, '')
	sentence = re.sub('[ ]{2,}', ' ',sentence)
	# print('---------------------------------')
	# print('??',str(len(sentence)), sentence)
	return sentence

# 輸入一句子，輸出 [[100, 204, '[**Name(Fuck)**]'], [133, 405, '[**1993-12-19**]'], ....]
def get_square_bracket_block(sentence):
	sbblock_index_list_list = []
	# print(sentence)
	start, end = 0, 0
	for charindex in range(0, len(sentence)):
		if sentence[charindex] == '[' and sentence[charindex+1] == '*' and sentence[charindex+2] == '*':
			start = charindex
			# print('start:', str(start))
		if charindex >= 2 and sentence[charindex-2] == '*' and sentence[charindex-1] == '*' and sentence[charindex] == ']':
			end = charindex
			# print('start:', str(start), 'end:', str(end+1), 'sentence[start, end]', sentence[start: end+1])
			sbblock_index_list_list.append([start, end+1, sentence[start:end+1]])
	# print(sbblock_index_list_list)
	return sbblock_index_list_list


def process_w2v_corpus(noteevent_dir_path, num_of_file):
	# with open('../data/w2v_corpus/processed_noteevent/' + file_name, 'r') as fi:
	context_list = []
	file_name_list = [str(f) for f in listdir(noteevent_dir_path) if '.txt' in str(f)]
	end = num_of_file
	file_name_list = file_name_list[0:end]
	
	i = 0
	for file_name in file_name_list:
		context = ''
		count = 1
		with open(noteevent_dir_path + file_name, 'r') as fi:
			for line in fi:
				if count > 2 and not line.isspace():
					line = remove_noise(line)
					# line = re.sub('[ ]+', ' ', line)
					context = context + line[:-1] + ' '
				count = count + 1
				
			context_list.append(context)
		
		print (file_name, i,'th is done.')
		i = i + 1
			
	with open('../../../data/w2v_corpus/NOTEEVENTS_' + str(end), 'w') as fo:
		for context in context_list:
			fo.write(context + '\n')



# input: NOTEEVENT1000
# output: NOTEEVENT1000.model
def produce_w2v_model(corpus_dir_path, corpus_file_name):
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	# # read file
	# sentences = word2vec.Text8Corpus(corpus_dir_path + corpus_file_name)

	# sentences = []
	# with open(corpus_dir_path + corpus_file_name, 'r') as fi:
	# 	for line in fi:
	# 		doc_as_sentence = [x.strip() for x in line.split(' ')]
	# 		sentences.append(doc_as_sentence)

	sentences = word2vec.LineSentence(corpus_dir_path + corpus_file_name)
	# # train the skip-gram model, defult window = 5
	# size: word vector dimen (num of features)
	# workers: numver of threads to run in parallel
	model = word2vec.Word2Vec(sentences, size=200, window=5, min_count=5, workers=4)
	# # save model
	model.save(corpus_dir_path + corpus_file_name + '.model')



def read_w2v_model(model_file_path, model_file_name):
	# load model
	model = word2vec.Word2Vec.load(model_file_path + model_file_name + '.model')
	return model


def write_centroid_map(dict, file_path, file_name):
	w = csv.writer(open(str(file_path + file_name) + ".csv", "w"))
	for key, val in dict.items():
		w.writerow([key, val])



# input: model, # of word per cluster
# output: dict{word, cluster}
def perform_w2v_clustering(model, word_per_cluster):
	start = time.time()
	# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an average of 5 words per cluster
	word_vectors = model.syn0
	num_clusters = word_vectors.shape[0] / word_per_cluster
	# print("word_vectors.shape[0]:" + str(word_vectors.shape[0]))
	
	# Initalize a k-means object and use it to extract centroids
	kmeans_clustering = KMeans( n_clusters = int(num_clusters) )
	idx = kmeans_clustering.fit_predict( word_vectors )

	# Create a Word / Index dictionary, mapping each vocabulary word to a cluster number                                                                                            
	word_centroid_map = dict(zip( model.index2word, idx ))

	# Get the end time and print how long the process took
	end = time.time()
	elapsed = end - start
	print ("Time taken for K Means clustering: ", elapsed, "seconds.")
	# print(word_centroid_map)
	return word_centroid_map

# 輸入 dict(word_centroid_map)和某個word
# 輸出一個cluster
def lookup_word_centroid_map(word_centroid_map ,word):
	cluster = ''
	if word in word_centroid_map:
		cluster = 'CLST_' + str(word_centroid_map[word])
	else:
		if word in string.punctuation:
			cluster = 'CLST_-1'
		else:
			cluster = 'CLST_-2'

	return cluster





def java_dep(doc_feature_list, sml_folder_path, file_name):
	with open(sml_folder_path + file_name + '.sml', 'r') as fi:
		sml_list = []
		for line in fi:
			if len(line) > 1:
				line_split = line[:-1].split('^^')
				word, pos, dep, parent = line_split[0], line_split[1], line_split[2], line_split[3]
				sml_list.append({'WORD':word, 'POS':pos, 'DEP':dep, 'PARENT':parent})

		cur_i = 0
		# for each sentence in a doc
		for s in range(0, len(doc_feature_list)):
			# for each word in a sentence
			for w in range(0, len(doc_feature_list[s])):
				# doc_feature_list[s][w] = ['The', {...}]
				word = doc_feature_list[s][w][0]
				for i in range(cur_i, len(sml_list)):
					# metamap_dict_list[i] = {'WORD': 'The', 'CID':'CUD-LESS', 'SEMTYPE':....}
					if word == sml_list[i]['WORD']:
						doc_feature_list[s][w][1]['Dependency'] = sml_list[i]['DEP']
						doc_feature_list[s][w][1]['ParentWord'] = sml_list[i]['PARENT']
						cur_i = i + 1
						break
					else:
						# 到最後一個都mapping不到
						if i == len(sml_list) - 1:
							doc_feature_list[s][w][1]['CID'] = u'CID-LESS'
							doc_feature_list[s][w][1]['SEMTYPE'] = u'SEMTYPE-LESS'
							doc_feature_list[s][w][1]['MATCH'] = u'MATCH-LESS'



def write_feature_file(stanford_folder_path, file_name, doc_feature_list):
	with open(stanford_folder_path + file_name + '.stanford', 'w') as fo:
		for sentence in doc_feature_list:
			if sentence == []:
				fo.write('\n')
			else:			
				for word in sentence:
					fo.write(word[0] + ' ')
					# fo.write(word[1]['CharacterOffsetBegin'] + ' ')
					# fo.write(word[1]['CharacterOffsetEnd'] + ' ')
					# fo.write(word[1]['WordLength'] + ' ')
					# fo.write(word[1]['Lemma'] + ' ')
					fo.write(word[1]['PartOfSpeech'] + ' ')
					fo.write(word[1]['CID'] + ' ')
					fo.write(word[1]['SEMTYPE'] + ' ')
					fo.write(word[1]['MATCH'] + ' ')
					fo.write(word[1]['Dependency'] + ' ')
					fo.write(word[1]['ParentWord'] + ' ')
					# fo.write(word[1]['ParentPOS'] + ' ')
					fo.write(word[1]['W2V'] + ' ')
					fo.write(word[1]['NamedEntityTag'] + '\n')
				fo.write('\n')



def alert():
	sys.stdout.write('\a')
	sys.stdout.write('\a')
	sys.stdout.write('\a')
	sys.stdout.flush()

































# # -*- coding: utf8 -*-
# from os import listdir
# import string
# import jsonrpclib
# from gensim.models import word2vec
# import time
# from sklearn.cluster import KMeans


# # 讀檔名
# def get_filename_list(folder_path):
# 	file_name_list = [f.split('.')[0] for f in listdir(folder_path) if '.DS_Store' not in str(f)]
# 	return file_name_list

# # 把 [u'The', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd'...}] 的dict多加depedency 和 parent的 tag
# def zip_indexeddepen_and_words(indexeddependencies, words):
# 	# 用單字的位置當key由小到大來sort indexeddep
# 	sorted_indexeddependencies = sorted(indexeddependencies, key=lambda k: int(k[-1].rsplit('-', 1)[-1]))
# 	# sorted_indexeddependencies = [
# 	# 								[u'root', u'ROOT-0', u'female-6'], 
# 	# 								[u'det', u'patient-2', u'The-1'], 
# 	# 								...
# 	# 								[u'conj_and', u'headache-10', u'dizziness-12']
# 	# 							   ]
# 	# words = [
# 	# 			[u'The', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'3', u'CharacterOffsetBegin': u'0', u'PartOfSpeech': u'DT', u'Lemma': u'the'}], 
# 	# 			[u'patient', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'11', u'CharacterOffsetBegin': u'4', u'PartOfSpeech': u'NN', u'Lemma': u'patient'}], 
# 	# 			...
# 	# 			[u'headache', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'63', u'CharacterOffsetBegin': u'55', u'PartOfSpeech': u'NN', u'Lemma': u'headache'}], [u'and', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'67', u'CharacterOffsetBegin': u'64', u'PartOfSpeech': u'CC', u'Lemma': u'and'}], [u'dizziness', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'77', u'CharacterOffsetBegin': u'68', u'PartOfSpeech': u'NN', u'Lemma': u'dizziness'}], 
# 	# 			[u'.', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'78', u'CharacterOffsetBegin': u'77', u'PartOfSpeech': u'.', u'Lemma': u'.'}]
# 	# 		  ]
# 	sorted_indexeddependencies_dict = {}
# 	for sortdep in sorted_indexeddependencies:
# 		# sortdep ： [u'root', u'ROOT-0', u'female-6']
# 		sorted_indexeddependencies_dict[int(sortdep[-1].rsplit('-', 1)[-1])] = sortdep[:-1]
	
# 	# sorted_indexeddependencies_dict = {1: [u'det', u'patient-2'], 2: [u'nsubj', u'female-6'], 3: [u'cop', u'female-6'], 4: [u'det', u'female-6'], 5: [u'amod', u'female-6'], 6: [u'root', u'ROOT-0'], 8: [u'prep_with', u'female-6'], 10: [u'prep_of', u'complaints-8'], 12: [u'conj_and', u'headache-10']}
# 	# print(d)

# 	for i in range(0, len(words)):
# 		if i+1 in sorted_indexeddependencies_dict:
# 			# print(str(i+1) +' in d')
# 			# print(sorted_indexeddependencies_dict[i+1][0], sorted_indexeddependencies_dict[i+1][1])
# 			# print(words[i][0], words[i][1])
# 			# words[i]: [u'The', {u'NamedEntityTag': u'O', u'Lemma': u'the'}]
# 			# words[i][1]: words[i]的dict裡多增加兩種key/val: Dependency, ParentWord
# 			words[i][1][u'Dependency'] = sorted_indexeddependencies_dict[i+1][0]
# 			words[i][1][u'ParentWord'] = sorted_indexeddependencies_dict[i+1][1].rsplit('-',1)[0]
# 			# words[i][1][u'ParentIndex'] = sorted_indexeddependencies_dict[i+1][1].rsplit('-',1)[1]
# 			if words[i][1][u'ParentWord'] == 'ROOT':
# 				words[i][1][u'ParentPOS'] = u'NO_PARENT_POS'
# 			else:
# 				parent_index = int(sorted_indexeddependencies_dict[i+1][1].rsplit('-',1)[1]) - 1
# 				words[i][1][u'ParentPOS'] = words[parent_index][1][u'PartOfSpeech']
# 		else:
# 			# 標點符號
# 			if words[i][0] in string.punctuation or words[i][0] in ['-LRB-','-RRB-','-LSB-','-RSB-','-LCB-','-RCB-']:
# 				words[i][1][u'Dependency'] = u'NO_DEP'
# 				words[i][1][u'ParentWord'] = u'NO_PARENT_WORD'
# 				words[i][1][u'ParentPOS'] = u'NO_PARENT_POS'
# 			else:
# 				words[i][1][u'Dependency'] = u'COLLAPSE'
# 				words[i][1][u'ParentWord'] = u'NO_PARENT_WORD'
# 				words[i][1][u'ParentPOS'] = u'NO_PARENT_POS'
# 	# print(words)

# 	return words
# # loads(server.parse(line))的時候把所有空格都忽略掉了
# # 要補len(trim掉的空白)跟len(prev_contect)回charoffset
# # input1 : words_with_dep = [ ['The', {'POS': 'NN', 'Dep': 'nsubj'...}], ['Patient', {...}]  ]
# # input2 : line = 目前讀進來的那一行
# # input3 : prev_context
# def shift_charoffset_in_words_with_depend(words_with_dep, line, prev_context):
# 	# left_trimmed_space_length = 0
# 	# if len(line) > 1 and not line.isspace():
# 	left_trimmed_space_length = len(line) - len(line.lstrip())

# 	# fo.write(str(int(feature_dict['CharacterOffsetBegin']) + len(prev_context) + left_trimmed_space_length) + ' ')
# 	# word: 'The'
# 	# feature_dict = {'POS': 'NN', 'Dep': 'nsubj'...}
# 	for i in range(0, len(words_with_dep)):
# 		word, feature_dict = words_with_dep[i][0], words_with_dep[i][1]
# 		# print('space=' + str(left_trimmed_space_length) + '     feature_dict[\'CharacterOffsetBegin\']:' + feature_dict['CharacterOffsetBegin'] + '    feature_dict[\'CharacterOffsetEnd\']:' + feature_dict['CharacterOffsetEnd'] + '   prev:' + str(len(prev_context)))
# 		newBegin = int(feature_dict['CharacterOffsetBegin']) + len(prev_context) + left_trimmed_space_length
# 		newEnd = int(feature_dict['CharacterOffsetEnd']) + len(prev_context) + left_trimmed_space_length
# 		words_with_dep[i][1]['CharacterOffsetBegin'] = str(newBegin)
# 		words_with_dep[i][1]['CharacterOffsetEnd'] = str(newEnd)
# 		words_with_dep[i][1]['WordLength'] = str(newEnd-newBegin)
# 		# words_with_dep[i][1]['CharacterOffsetBegin'] = unicode(str(newBegin), 'utf-8')
# 		# words_with_dep[i][1]['CharacterOffsetEnd'] = unicode(str(newEnd), 'utf-8')
# 		# words_with_dep[i][1]['WordLength'] = unicode(str(newEnd-newBegin), 'utf-8')
# 	return words_with_dep	





# # input: NOTEEVENT1000
# # output: NOTEEVENT1000.model
# def produce_w2v_model(corpus_dir_path, corpus_file_name):
# 	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 	# # read file
# 	# sentences = word2vec.Text8Corpus(corpus_dir_path + corpus_file_name)

# 	sentences = []
# 	with open(corpus_dir_path + corpus_file_name, 'r') as fi:
# 		for line in fi:
# 			doc_as_sentence = [x.strip() for x in line.split()]
# 			sentences.append(doc_as_sentence)

# 	# # train the skip-gram model, defult window = 5
# 	# size: word vector dimen (num of features)
# 	# workers: numver of threads to run in parallel
# 	model = word2vec.Word2Vec(sentences, size=200, window=5, min_count=5)
# 	# # save model
# 	model.save(corpus_dir_path + corpus_file_name + '.model')



# def read_w2v_model(model_file_name):
# 	# load model
# 	model = word2vec.Word2Vec.load(model_file_name)
# 	return model


# def write_centroid_map(dict, file_name):
# 	w = csv.writer(open(str(file_name) + ".csv", "w"))
# 	for key, val in dict.items():
# 		w.writerow([key, val])



# # input: model, # of word per cluster
# # output: dict{word, cluster}
# def perform_w2v_clustering(model, word_per_cluster):
# 	start = time.time()
# 	# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an average of 5 words per cluster
# 	word_vectors = model.syn0
# 	num_clusters = word_vectors.shape[0] / word_per_cluster
# 	# print("word_vectors.shape[0]:" + str(word_vectors.shape[0]))
	
# 	# Initalize a k-means object and use it to extract centroids
# 	kmeans_clustering = KMeans( n_clusters = int(num_clusters) )
# 	idx = kmeans_clustering.fit_predict( word_vectors )

# 	# Create a Word / Index dictionary, mapping each vocabulary word to a cluster number                                                                                            
# 	word_centroid_map = dict(zip( model.index2word, idx ))

# 	# Get the end time and print how long the process took
# 	end = time.time()
# 	elapsed = end - start
# 	print ("Time taken for K Means clustering: ", elapsed, "seconds.")
# 	# print(word_centroid_map)
# 	return word_centroid_map

# # 輸入 dict(word_centroid_map)和某個word
# # 輸出一個cluster
# def lookup_word_centroid_map(word_centroid_map ,word):
# 	cluster = ''
# 	if word in word_centroid_map:
# 		cluster = 'CLST_' + str(word_centroid_map[word])
# 	else:
# 		if word in string.punctuation:
# 			cluster = 'CLST_-1'
# 		else:
# 			cluster = 'CLST_-2'

# 	return cluster


# def write_feature_file(stanford_folder_path, file_name, doc_feature_list):
# 	with open(stanford_folder_path + file_name + '.stanford', 'w') as fo:
# 		for sentence in doc_feature_list:
# 			if sentence == []:
# 				fo.write('\n')
# 			else:			
# 				for word in sentence:
# 					fo.write(word[0] + ' ')
# 					# fo.write(word[1]['CharacterOffsetBegin'] + ' ')
# 					# fo.write(word[1]['CharacterOffsetEnd'] + ' ')
# 					fo.write(word[1]['WordLength'] + ' ')
# 					fo.write(word[1]['Lemma'] + ' ')
# 					fo.write(word[1]['PartOfSpeech'] + ' ')
# 					fo.write(word[1]['CID'] + ' ')
# 					fo.write(word[1]['SEMTYPE'] + ' ')
# 					fo.write(word[1]['MATCH'] + ' ')
# 					fo.write(word[1]['Dependency'] + ' ')
# 					fo.write(word[1]['ParentWord'] + ' ')
# 					# fo.write(word[1]['ParentPOS'] + ' ')
# 					# fo.write(word[1]['W2V'] + ' ')
# 					fo.write(word[1]['NamedEntityTag'] + '\n')
# 				fo.write('\n')
