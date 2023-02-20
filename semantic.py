# ***********************************************************************************#
#                             SIMILARITY WITH SPACY                                  #
# ***********************************************************************************#

#  Run  python -m spacy download en_core_web_md in the command line to download the advance english model
# working the the english advanced model 'en_core_web_md'

import spacy
nlp = spacy.load('en_core_web_md')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

'output'
'''
0.5929929675536907  
0.40415016164997786 
0.22358825939615987
'''
# Note:
# The value 0.5929929675536907 indicates that "cat" and "monkey" are somewhat similar, as both are animals ( higheest )
# he value 0.40415016164997786 indicates that "banana" and "monkey" are somewhat dissimilar, as they are different types of objects but still shows considerable similarities as monkey eats banana
# The value 0.22358825939615987 indicates that "banana" and "cat" are even less similar  than "banana" and "monkey" as there are 2 differents objects and cat do not eat banana
# The similarity score is based on the context and meaning of the words, rather than just their spelling or pronunciation
print("\n")


# ---  WORKING WITH VECTORS ----
# similarities in words  "cat apple monkey banana"

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

'output'

'''
cat cat 1.0
cat apple 0.20368057489395142
cat monkey 0.5929929614067078
cat banana 0.2235882580280304
apple cat 0.20368057489395142
apple apple 1.0
apple monkey 0.2342509925365448
apple banana 0.6646699905395508
monkey cat 0.5929929614067078
monkey apple 0.2342509925365448
monkey monkey 1.0
monkey banana 0.4041501581668854
banana cat 0.2235882580280304
banana apple 0.6646699905395508
banana monkey 0.4041501581668854
banana banana 1.0

'''
# Note: The similarity between a token and itself is always 1.0
# we can notice that some pairs have higher similarity score like monkey and cat as animals, apple  and banana as fruits,
# and some have the lowest score like cat and apple, cat and banana, apple and monkey which have less similarities as fruit and animals that do not eat these fruit. for example cat do not have any significant  similarities with any of the fruits
# but we can notice monkey and banana have more similarity with the fruits than the cat


print("\n")

# WORKING WITH SENTENCES
# --- similarities beetwen longeur  sentences ----
sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


'output'

'''
where did my dog go -  0.630065230699739
Hello, there is my car -  0.8033180111627156
I've lost my car in my car -  0.6787541571030323
I'd like my boat back -  0.562494104588661
I will name my dog Diana -  0.6491444051802615

'''

# The output shows that the similarity scores range from 0.56 to 0.80, there is a strong similarities between them as there are all complaints
# with the highest score being 0.80 for the sentence "Hello, there is my car" as refers to a car
#The output shows that the sentence "Hello, there is my car" is the most similar to the model sentence, with a similarity score of 0.8033180111627156.
# next we have  "where did my dog go" and "I've lost my car in my car", with similarity scores of 0.630065230699739 and 0.6787541571030323, respectively.
# The leat similar to the reference sentence are 'I'd like my boat back' and 'I will name my dog Diana ' 


## ====SAME EXAMPLE BUT NOW WORKING WITH THE SIMPLER LANGUAGE MODEL 'en_core_web_sm'=======

import spacy
nlp = spacy.load('en_core_web_sm')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

'output '

'''
UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger, parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models instead if available.
  print(word1.similarity(word2))
0.6770567131180597
UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger, parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models instead if available.
  print(word3.similarity(word2))
0.7276310914874259
UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger, parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models instead if available.
  print(word3.similarity(word1))
0.6806929608512433

'''
# the reults quite different from the previous model cat is vey similar to monkey as animals 
# as we can notice in the  third print banana is pretty much similart to cat with a score of 0.6806929608512433 which is biased. not accurate


# ---  WORKING WITH VECTORS ----
# similarities in words  "cat apple monkey banana"

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
        
'output'

'''
UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Token.similarity method will be based on the tagger, parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models instead if available.
  print(token1.text, token2.text, token1.similarity(token2))
cat apple 0.7018380165100098
cat monkey 0.6455237865447998
cat banana 0.22147215902805328
apple cat 0.7018380165100098
apple apple 1.0
apple monkey 0.7389945983886719
apple banana 0.3619703948497772
monkey cat 0.6455237865447998
monkey apple 0.7389945983886719
monkey monkey 1.0
monkey banana 0.42320212721824646
banana cat 0.22147215902805328
banana apple 0.3619703948497772
banana monkey 0.42320212721824646
banana banana 1.0

'''
# here we have high score for cat and apple(cat apple 0.7018380165100098) , apple and monkey (apple monkey 0.7389945983886719)
# previously  low in the 'en_core_web_md' model (cat apple 0.20368057489395142) and (apple monkey 0.2342509925365448) which is not accurate.


# WORKING WITH SENTENCES
# --- similarities beetwen longeur  sentences ----
sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


'output'

'''
UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger, parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models instead if available.
  similarity = nlp(sentence).similarity(model_sentence)
where did my dog go -  0.4043350835838601
Hello, there is my car -  0.5648939509223623
I've lost my car in my car -  0.5480285339200293
I'd like my boat back -  0.3007498554479068
I will name my dog Diana -  0.39040753562783626

'''

# Note for the use of the simpler english model
 #The warning message is indicating that the model being used (en_core_web_sm)
 # does not have pre-trained word vectors, and as a result, 
 # the similarity calculation will only be based on context which may not provide useful similarity judgments. as we can notice here
 # for the same code with have differents results with reduce similarity score as this model is less accurate
 # as demontrated in the above codes