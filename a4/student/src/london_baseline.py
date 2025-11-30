# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    argp = argparse.ArgumentParser()
    argp.add_argument('--eval_corpus_path', default=None)
    args = argp.parse_args()
    predictions = ["London" for i in open(args.eval_corpus_path, encoding='utf-8') ]
    total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    accuracy = correct / total
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
