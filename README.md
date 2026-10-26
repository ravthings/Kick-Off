# Kick Off ⚽️
A small language model that writes football text, built from scratch in PyTorch.
It's trained on the Wikipedia pages of four clubs (FC Barcelona, Manchester City, Liverpool and Real Madrid) and learns to predict the next word.

## How it Works
1. **Data**: downloads the four Wikipedia pages and saves them to `football.txt` (only the first time it runs).
2. **Tokenizer**: splits the text into words and gives each unique word a number. The vocabulary is about [7,350] words.
3. **Batches**: picks random 64-word chunks from the text. The target for each word is the word that comes after it.
4. **Model**: a word-level bigram model. It's one embedding table that gives a score for every possible next word, based on the current word.
5. **Training**: cross-entropy loss and the AdamW optimizer (learning rate 0.01). The model saves to `kickoff.pth` and carries on from there next time.
6. **Generation**: starts from a word like "Liverpool" and keeps sampling the next word.

## How to run?
`pip install torch wikipedia-api`

`python main.py`

The script asks `Do you want to train the model? y/n`:

`y` trains for 1000 steps (carrying on from the saved model if there is one), saves, and writes some text.

`n` loads the saved model, shows the validation loss, and writes some text.

## Results
https://github.com/user-attachments/assets/5e805c81-f033-4c74-ad68-e9b7425f2860

The model overfitted

## Limitations
* It only looks at **one** word to guess the next, so pairs of words make sense but whole sentences don't.
* The dataset is small (about 38,000 words).
* Punctuation stays attached to words, so "City" and "City." count as different words.
