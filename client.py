#-*- coding: utf-8 -*-　　 
# https://github.com/dasmith/stanford-corenlp-python
# from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp
import jsonrpclib
from pprint import pprint
import pickle
from simplejson import loads
import os.path
import time, string
import datetime
import csv
from myfunctions import get_filename_list
from myfunctions import zip_indexeddepen_and_words, shift_charoffset_in_words_with_depend
from myfunctions import process_w2v_corpus, produce_w2v_model, read_w2v_model, perform_w2v_clustering, write_centroid_map, lookup_word_centroid_map
from myfunctions import write_feature_file



if __name__ == '__main__':
	server = jsonrpclib.Server("http://localhost:8080")
	text1_folder_path = '../../../data/train/discharge/text1/'
	stanford_folder_path = '../../../data/train/discharge/stanford/'
	bieso_folder_path = '../../../data/train/discharge/bieso/'
	metamap_folder_path = '../../../data/train/discharge/metamap_0127/'

	noteevent_folder_path = '../../../data/w2v_corpus/processed_noteevent/'
	corpus_dir_path = '../../../data/w2v_corpus/'
	corpus_file_name  = 'NOTEEVENTS_1000'

	w2v_model_path = '../../../data/w2v_corpus/'
	w2v_model_name = 'NOTEEVENTS_1000'
	
	file_name_list = get_filename_list(text1_folder_path)
	file_name_list.remove('14708-006815')
	file_name_list.remove('21413-012450')
	# file_name_list = ['02652-006395']

# 產生，讀取w2v model

	# 流程：1. process_w2v_corpus()：處裡一堆noteevent，產生一個corpus檔
	#      2.1 produce_w2v_model()： 吃corpus檔，產生w2v_model
	# 	   2.2 read_w2v_model()：    吃model路徑和檔名，產生w2v_model
	#      3. perfrom_clustering()： 吃model做cluster，產出centroid map 
	#      4. (Optional) write_centroid_map()： 寫入centroid map成csv

	# 1.
	# process_w2v_corpus(noteevent_folder_path, 1000)

	# 2.1
	# produce_w2v_model(corpus_dir_path, corpus_file_name)
	# 2.2
	w2v_model = read_w2v_model(w2v_model_path, w2v_model_name)

	# 3.
	# word_centroid_map = perform_w2v_clustering(w2v_model, 7129)
	word_centroid_map = perform_w2v_clustering(w2v_model, 200)
	
	# 4.
	# write_centroid_map(word_centroid_map, corpus_dir_path, str(corpus_file_name))

	# print(model)


	# file_name_list = ['10644-007491']
	for file_name in file_name_list:
		t_start = datetime.datetime.now()
		print (file_name + ' starts at...' + str(t_start))
# 用stanford-corenlp產生pos, lemma, dep, ...等feature 
		# 存整篇文章的stanford feature
		# doc_feature_list = [ line1的offset_words_with_dep, line2的offset_words_with_dep, ... ]
		doc_feature_list = []  
		prev_context = ''
		with open(text1_folder_path + file_name + '.text.txt1', 'r') as fi:	
			for line in fi:
				if len(line) > 1 and not line.isspace():
					# line丟進去處裡, parse出一堆sentences
					parsed_sentences_in_line = loads(server.parse(line))
					# fo.write(line+'\n\n')
					for parsed_sentence_in_line in parsed_sentences_in_line['sentences']:
						# parsed_sentence_in_line['indexeddependencies'], parsed_sentence_in_line['parsetree'], parsed_sentence_in_line['words']
						# 把dep給zip到原本的feature上
						words_with_dep = zip_indexeddepen_and_words(parsed_sentence_in_line['indexeddependencies'], parsed_sentence_in_line['words'])

						offset_words_with_dep = shift_charoffset_in_words_with_depend(words_with_dep, line, prev_context)
						doc_feature_list.append(offset_words_with_dep)					
				else:
					doc_feature_list.append([])
				prev_context += line

#讀取bieso檔，並和stanford feature (doc_feature_list) 組起來
		# read bieso
		bieso_dict_list = []
		with open(bieso_folder_path + file_name + '.bieso', 'r') as fi2:
			for line in fi2:
				if len(line) > 1:
					# line = '121~~~126~~~Chest~~~B'
					line_list = line[:-1].split('~~~')
					s_index, e_index, word, bieso = line_list[0], line_list[1], line_list[2], line_list[3]
					bieso_dict_list.append({'s_index':s_index, 'e_index':e_index, 'word':word, 'bieso':bieso})
		# print('bieso_dict_list', bieso_dict_list)

		# 把bieso和feature組起來
		cur_i = 0
		# doc_feature_list中的每個sentence
		for s in range(0, len(doc_feature_list)):
			# for each word in a sentence
			for w in range(0, len(doc_feature_list[s])):
				word, s_index, e_index = doc_feature_list[s][w][0], doc_feature_list[s][w][1]['CharacterOffsetBegin'], doc_feature_list[s][w][1]['CharacterOffsetEnd']
				for i in range(cur_i, len(bieso_dict_list)):
					# print(word + '     bieso_dict_list[i][\'word\']:' + bieso_dict_list[i]['word'] + '    s_index:' + str(s_index) + '    bieso_dict_list[i][\'s_index\']:' + bieso_dict_list[i]['s_index']+ '  e_index:' + e_index + '  bieso_dict_list[i][\'e_index\']:' + bieso_dict_list[i]['e_index'])
					if word == bieso_dict_list[i]['word'] and s_index == bieso_dict_list[i]['s_index'] and e_index == bieso_dict_list[i]['e_index']:
						doc_feature_list[s][w][1]['NamedEntityTag'] = bieso_dict_list[i]['bieso']
						# print(word, doc_feature_list[s][w][1]['NamedEntityTag'])
						cur_i = i + 1
						break
					else:
						if i == len(bieso_dict_list) - 1:
							doc_feature_list[s][w][1]['NamedEntityTag'] = u'O'
							# print('             ', word, 'O')
				doc_feature_list[s][w][1]['W2V'] = lookup_word_centroid_map(word_centroid_map, word)


#讀取metamap檔，並和含stanford feature的doc_feature_list組起來
		# read metamap
		metamap_dict_list = []
		with open(metamap_folder_path + file_name + '.metamap', 'r') as fi3:
			for line in fi3:
				if len(line) > 1:
					line_list = line[:-1].split(' ')
					word, cID, semtype, match = line_list[0], line_list[1], line_list[2], line_list[3]
					tmp = {'(':'-LRB-', ')':'-RRB-', '[':'-LSB-', ']':'-RSB-', '{':'-LCB-', '}': '-RCB-'}
					if word in tmp:
						metamap_dict_list.append({'WORD':tmp[word], 'CID':cID, 'SEMTYPE':semtype, 'MATCH':match})
					else:
						metamap_dict_list.append({'WORD':word, 'CID':cID, 'SEMTYPE':semtype, 'MATCH':match})

		#把metamap和feature組起來		
		cur_i = 0
		# for each sentence in a doc
		for s in range(0, len(doc_feature_list)):
			# for each word in a sentence
			for w in range(0, len(doc_feature_list[s])):
				# doc_feature_list[s][w] = ['The', {...}]
				word = doc_feature_list[s][w][0]
				for i in range(cur_i, len(metamap_dict_list)):
					# metamap_dict_list[i] = {'WORD': 'The', 'CID':'CUD-LESS', 'SEMTYPE':....}
					if word == metamap_dict_list[i]['WORD']:
						doc_feature_list[s][w][1]['CID'] = metamap_dict_list[i]['CID']
						doc_feature_list[s][w][1]['SEMTYPE'] = metamap_dict_list[i]['SEMTYPE']
						doc_feature_list[s][w][1]['MATCH'] = metamap_dict_list[i]['MATCH']
						cur_i = i + 1
						break
					else:
						# 到最後一個都mapping不到
						if i == len(metamap_dict_list) - 1:
							doc_feature_list[s][w][1]['CID'] = u'CID-LESS'
							doc_feature_list[s][w][1]['SEMTYPE'] = u'SEMTYPE-LESS'
							doc_feature_list[s][w][1]['MATCH'] = u'MATCH-LESS'
# 對照w2v每個word的cluster
				doc_feature_list[s][w][1]['W2V'] = lookup_word_centroid_map(word_centroid_map, word)
			

# 讀取sml拼起來
		sml_folder_path = '../../../data/train/discharge/sml_02013/'

		# doc_feature_list = java_dep(doc_feature_list, sml_folder_path, file_name)


# write feature files
		write_feature_file(stanford_folder_path, file_name, doc_feature_list)

		end_time = datetime.datetime.now()
		print ('\tends at.....' + str(end_time) , 'costs ' + str(end_time-t_start) )

# 結束通知
	alert()
