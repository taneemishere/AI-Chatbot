import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from termcolor import colored

import nltk
from nltk.tokenize import sent_tokenize

# The NLTK Downloads we need
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('nps_chat')


# Global Constants
GREETING_INPUTS = ("hello", "hi")
GREETING_RESPONCES = ["hi", "hey", "*nods*", "hi there", "Talkin' to me?"]
FILE_NAME = "faq.txt"

# Global Variables
lem = nltk.stem.WordNetLemmatizer()
remove_punctuation = dict((ord(punct), None) for punct in string.punctuation)


# Functions


def fetch_features(chat):
    # fetch_features() transforms a chat into a classifier friendly format

    features = {}
    for word in nltk.word_tokenize(chat):
        features['contains({})'.format(word.lower())] = True
    return features


def lemmatise(tokens):
    # This method performs the lemmatization on the words
    return [lem.lemmatize(token) for token in tokens]


def tokenise(text):
    # This method tokenizes the words
    return lemmatise(nltk.word_tokenize(text.lower().translate(remove_punctuation)))


def greet(sentence):
    # This method responses with a standard a bot recognizes
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONCES)


def match(user_responce):
    # This method match matches a user input to the existing set of questions
    resp = ''
    q_list.append(user_responce)
    TfidfVec = TfidfVectorizer(tokenizer=tokenise, stop_words='english')

    tfidf = TfidfVec.fit_transform(q_list)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()

    req_tfidf = flat[-2]

    if req_tfidf == 0:
        resp = resp + "Sorry! I don't know the answer to this. Would you like to try again? Type Ciao to exit"
        return resp

    else:
        resp_ids = qa_dict[idx]
        resp_str = ''
        s_id = resp_ids[0]
        end = resp_ids[1]

        while s_id < end:
            resp_str = resp_str + " " + sent_tokenize[s_id]
            s_id += 1

        resp = resp + resp_str
        return resp


# Training the classifier

# Fetching the chat corpus
chats = nltk.corpus.nps_chat.xml_posts()[:10000]

# Extract the features from chat
featuresets = [(fetch_features(chat.text), chat.get('class')) for chat in chats]

# splitting into test and train sets
size = int(len(featuresets) * 0.1)  # 10%
train_set, test_set = featuresets[size:], featuresets[:size]  # 90 training and 10 testing

classifier = nltk.MaxentClassifier.train(train_set)
# for NaiveBayesClassifier
# classifier = nltk.NaiveBayesClassifier.train(train_set)

# print the accuracy final one
print(nltk.classify.accuracy(classifier, test_set))

# Question Bank Creation
ques_bank = open(FILE_NAME, 'r', errors='ignore')

qb_text = ques_bank.read()
qb_text = qb_text.lower()

sent_tokens = nltk.sent_tokenize(qb_text)  # Converts the list into sentences
word_tokens = nltk.word_tokenize(qb_text)  # converts the list of words

qa_dict = {}  # The Dictionary to store questions and corresponding answers
q_list = []  # List of all questions
s_count = 0  # Sentence counter

# Extract questions and answers
# Answer is all the content between 2 questions [assumption]


while s_count < len(sent_tokens):
    result = classifier.classify(fetch_features(sent_tokens[s_count]))
    if "question" in result.lower():
        next_question_id = s_count + 1
        next_question = classifier.classify(fetch_features(sent_tokens[next_question_id]))

        while not ("question" in next_question.lower()) and next_question_id < len(sent_tokens) - 1:
            next_question_id += 1
            next_question = classifier.classify(fetch_features(sent_tokens[next_question_id]))

        q_list.append(sent_tokens[s_count])
        end = next_question_id

        if (next_question_id - s_count) > 5:
            end = s_count + 5

        qa_dict.update({len(q_list) - 1: [s_count + 1, end]})
        s_count = next_question_id

    else:
        s_count += 1

# Response Fetching
flag = True
print(colored("NEO: \nI'm a Neo, I have all the answers, if you want to exit, type Ciao", 'blue', attrs=['bold']))

while flag == True:
    print(colored("\nYOU: ", 'red', attrs=['bold']))
    u_input = input()
    u_input = u_input.lower()

    if u_input != 'ciao':
        if greet(u_input) is not None:
            print(colored("\nNEO:", 'blue', attrs=['bold']))
            print(greet(u_input))

        else:
            print(colored("\nNEO:", 'blue', attrs=['bold']))
            print(colored(match(u_input).strip().capitalize(), 'blue'))
            q_list.remove(u_input)

    else:
        flag = False
        print(colored("\nNEO: Bye! take care..", 'blue', attrs=['bold']))
