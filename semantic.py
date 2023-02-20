import spacy

# run all the code extracts above

nlp = spacy.load("en_core_web_md")

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

tokens = nlp("cat apple monkey banana ")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


sentence_to_compare = "Why is my cat on the car"
sentences = [
    "where did my dog go",
    "Hello, there is my car",
    "I've lost my car in my car",
    "I'd like my boat back",
    "I will name my dog Diana",
]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# Write a note about what you found interesting about the similarities between cat, monkey and banana and think of an example of your own.
# It's interesting to see that common biases are taken into account, for example monkeys wouldn't be so strongly associated with bananas over other fruit if not for common biases introduced through books and other media.
# Another unusual thing is how "cat" and "apple" are considered less similar than "cat" and "banana"

# ●Run the example file with the simpler language model ‘en_core_web_sm’ and write a note on what you notice is different from the model 'en_core_web_md'.
# The simple language model is less confident about the similarity between sentences, it's less able to recognize and relate as many different words at the larger model.

# ●Host your solution on GitHub. Ensure your repo includes a Dockerfile and README.md with instructions on how to run it.
# ○If it doesn’t already, please ensure that your repo includes a file named requirements.txt to automate the installation of the project’s requirements.
# ○Remember to exclude any venv or virtualenv files from your repo.
# ●Link to your public remote Git repo in a file named  semantic_similarity.txt
