
# 1. get tokenized data
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from data_chatbot import questions_answers,loadDF

from vocab_chatbot import Vocab
questions_train, questions_valid, answers_train, answers_valid = questions_answers()
for q,a in zip(questions_train[0],answers_train[0]):
    print(f"Question: {q}.\n Answer: {a}.")


# 2. build vocabularies

vQ = Vocab("Questions")
# run program and see the error caused by the next line!!!
for sequence in ["<SOS>", "<EOS>"] + questions_train + questions_valid:
    for token in sequence:
        vQ.indexWord(token)
print(len(list(vQ.words)))
print(list(vQ.words)[:5])
        
        
    