import argparse
import os
import glob
from collections import defaultdict
from split import load_dataset_from_column

# compares `original_test_filename` with `model_predicted_filename` under all paths, and merge the results
# paths could be like ['splitted_0/split-0', 'splitted_0/split-1', ..., 'splitted_1/split-0'...]
def load_from_splits(paths, original_test_filename, model_predicted_filename):
    sentence_potential_mistake_count = defaultdict(int)
    for path in paths:
        original_test = os.path.join(path, original_test_filename)
        model_predicted = os.path.join(path, model_predicted_filename)
        assert os.path.exists(original_test)
        assert os.path.exists(model_predicted)
        original_test = load_dataset_from_column(original_test)
        model_predicted = load_dataset_from_column(model_predicted)
        for (original_sentence, original_labels), (model_sentence, model_labels) in zip(original_test, model_predicted):
            assert ' '.join(original_sentence) == ' '.join(model_sentence)
            if ' '.join(original_labels) != ' '.join(model_labels):
                sentence_potential_mistake_count[' '.join(original_sentence)] += 1
    return sentence_potential_mistake_count

def form_weighted_train_set(train_file, eps, mistake_count):
    assert os.path.exists(train_file)
    train_set = load_dataset_from_column(train_file)
    weighted_train_set = []
    for sentence, labels in train_set:
        mistakes = mistake_count.get(' '.join(sentence), 0)
        weight = eps ** mistakes
        weighted_train_set.append([sentence, labels, [weight] * len(labels)])
    return weighted_train_set

def main(split_folders, train_file, output_weighted_train_file, model_predicted_filename, eps):
    for split_folder in split_folders:
        assert os.path.exists(split_folder)
    assert not os.path.exists(output_weighted_train_file)
    paths = []
    for split_folder in split_folders:
        paths.extend(glob.glob(os.path.join(split_folder, 'split-*')))
    sentence_potential_mistake_count = load_from_splits(paths, 'test.bio', model_predicted_filename)
    weighted_train_set = form_weighted_train_set(train_file, eps, sentence_potential_mistake_count)
    with open(output_weighted_train_file, 'w') as f:
        for sentence, labels, weights in weighted_train_set:
            for token, label, weight in zip(sentence, labels, weights):
                f.write(f'{token}\t{label}\t{weight}\n')
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_folders', nargs='+',required=True)
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model_predicted_filename', default='predict.bio')
    parser.add_argument('--eps', type=float, default=0.7)
    args = parser.parse_args()
    print(vars(args))
    main(args.split_folders, args.train_file, args.output, args.model_predicted_filename, args.eps)