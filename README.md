# Semantic similarity(NLP)
  Using spacy library to work on similarity in words, sentences and paragraphs and compare models


#  Projec Name
similarity-spacy

# A brief description of the project
This project  presents  how to use  Python Natural Language Processing library spaCy to work out similarities between words, sentences and paragraphs using two models  the  english simple model 'en_core_web_sm'  and the more advance model 'en_core_web_md' . compare and contrast the difference between theses models

 ## Installation

 - Clone  the repository
 
 git clone https://github.com/MaxxwellG/similarity-spacy.git

 -Install the required packages by running in the command prompt or twerminal and run:

pip install -r requirements.txt

or go to the root of the repository directory and run the pip  command to automatically  create the requirements.txt file:

pip freeze > requirements.txt

# Usage
This project can be used to analise similarities between words , paragraphs and sentences using 2 english models 'en_core_web_sm'  and 'en_core_web_md'.
the project helps understand the difference between the models

# ------------ Working with spaCy-------

# Before we proceed with spacy it must be installed
pip install spacy

# spaCy is an open-source natural language processing library that is designed to be fast and efficient.
# spaCy provides a wide range of functionalities for various natural language processing tasks such as tokenization, named entity recognition, dependency parsing, and more. This project will focus on finding  similarities and comparing the  different english  models
# Here's a simple example of how we can find similarities between , words, sentences using spacy

# import spacy(ony after installing spacy)
import spacy

# Downnload the simple English model
python3 -m spacy download en_core_web_sm

# Downnload the advanced  English model
python3 -m spacy download en_core_web_md

# load the english model
nlp = spacy.load('en_core_web_md')


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

 

 # Contributing 
 Contributions are welcome!

# Authors
Maxime HT
# Version History

    0.2
        Various bug fixes and optimizations
        See commit change or See release history
    0.1
        Initial Release

 # License
 Distributed under the GNU license. see 'LICENSE'  for more information.


