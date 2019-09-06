import sys
import os
import random
import json
import argparse


def validate_bio(labels):
    for cur_label, next_label in zip(labels, labels[1:] + ['O']):
        if cur_label[0] == 'O':
            assert next_label[0] == 'O' or next_label[0] == 'B'
            continue
        elif cur_label[0] == 'B':
            assert next_label[0] == 'O' or next_label[0] == 'B' or (
                        next_label[0] == 'I' and cur_label[1:] == next_label[1:])
        elif cur_label[0] == 'I':
            assert next_label[0] == 'O' or next_label[0] == 'B' or \
                   (next_label[0] == 'I' and cur_label[1:] == next_label[1:])
        else:
            assert False


def iob2bio(iob_labels):
    bio_labels = []
    for prev_label, cur_label in zip(['O'] + iob_labels[:-1], iob_labels):
        if (prev_label[0] == 'O' and cur_label[0] == 'I') or (prev_label[0] != 'O' and
                                                              cur_label[0] == 'I' and
                                                              prev_label[2:] != cur_label[2:]):
            bio_labels.append('B' + cur_label[1:])
        else:
            bio_labels.append(cur_label)
    return bio_labels

def iobes2bio(iobes_labels):
    bio_labels = []
    for label in iobes_labels:
        if label[0] == 'S':
            bio_labels.append('B' + label[1:])
        elif label[0] == 'E':
            bio_labels.append('I' + label[1:])
        else:
            bio_labels.append(label)
    return bio_labels


# loads a column dataset into list of (tokens, labels)
# assumes BIO(IOB2) labeling
def load_dataset_from_column(path, schema='bio'):
    with open(path, 'r', encoding='utf-8') as f:
        sentences = []
        tokens = []
        labels = []
        for line in f.readlines() + ['']:
            if len(line) == 0 or line.startswith('-DOCSTART-') or line.isspace():
                if len(tokens) > 0:
                    if schema is not None and schema != 'none':
                        if schema == 'iob':
                            labels = iob2bio(labels)
                        elif schema == 'iobes':
                            labels = iobes2bio(labels)
                        validate_bio(labels)
                    sentences.append((tokens, labels))
                tokens = []
                labels = []
            else:
                splits = line.strip().split()
                token, label = splits[0], splits[-1]
                tokens.append(token)
                labels.append(label)
    return sentences


# given tokens, labels, extract list of spans of entities as (TYPE, START inc, END exc, SURFACE)
def sent_label_to_entity(tokens, labels):
    assert len(tokens) == len(labels)
    entities = []
    cur_entity = {}
    for index, (label, next_label) in enumerate(zip(labels, labels[1:] + ['O'])):
        if label[0] == 'B':
            cur_entity['type'] = label[2:]
            cur_entity['start'] = index
        if next_label[0] != 'I' and len(cur_entity) > 0:
            cur_entity['end'] = index + 1
            cur_entity['surface'] = ' '.join(tokens[cur_entity['start']: cur_entity['end']])
            entities.append(cur_entity)
            cur_entity = {}
    return entities


# sentence_entities: [[e1, e2, e3], [e2, e4, e5], [e1, e3], ...]...
# folds: int
# random_seed: int
def create_folds(sentence_entities, folds, random_seed):
    random.seed(random_seed)
    data_size = len(sentence_entities)
    indexs = list(range(data_size))
    info = {'seed': random_seed, 'folds': folds, 'indexs': indexs}
    random.shuffle(indexs)
    for i in range(folds):
        test_data_indexs = indexs[i::folds]
        train_data_indexs = [indexs[x::folds] for x in range(folds) if x != i]
        train_data_indexs = [x for y in train_data_indexs for x in y]
        forbid_entities = set().union(*[set(sentence_entities[x]) for x in test_data_indexs])
        train_data_indexs = list(
            filter(lambda x: set(sentence_entities[x]).isdisjoint(forbid_entities), train_data_indexs))
        assert set(test_data_indexs).isdisjoint(set(train_data_indexs))
        assert set().union(*[set(sentence_entities[x]) for x in test_data_indexs]).isdisjoint(
            set().union(*[set(sentence_entities[x]) for x in train_data_indexs]))
        _info = {
            'train_indexs': train_data_indexs,
            'test_indexs': test_data_indexs,
            'train_sentences': len(train_data_indexs),
            'train_total_entities': sum(len(sentence_entities[x]) for x in train_data_indexs),
            'train_distinct_entities': len(set().union(*[set(sentence_entities[x]) for x in train_data_indexs])),
            'test_sentences': len(test_data_indexs),
            'test_total_entities': sum(len(sentence_entities[x]) for x in test_data_indexs),
            'test_distinct_entities': len(set().union(*[set(sentence_entities[x]) for x in test_data_indexs])),
        }
        info[f'fold-{i}'] = _info
        print(f"Set {i}")
        print(f"Train sentences: {_info['train_sentences']}")
        print(f"Train total entities: {_info['train_total_entities']}")
        print(f"Train distinct entities: {_info['train_distinct_entities']}")
        print(f"Test sentences: {_info['test_sentences']}")
        print(f"Test total entities: {_info['test_total_entities']}")
        print(f"Test distinct entities: {_info['test_distinct_entities']}")
    return info


def main(input_files, output_folder, folds, schema):
    if os.path.exists(output_folder):
        print(f"Output folder {output_folder} exists, exiting...")
        sys.exit(1)
    os.makedirs(output_folder, exist_ok=True)
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist, exiting...")
            sys.exit(1)
    assert folds > 0

    all_data = []
    for input_file in input_files:
        all_data.extend(load_dataset_from_column(input_file, schema))

    sentence_entities = [list(map(lambda x: x['surface'], sent_label_to_entity(tokens, labels)))
                         for tokens, labels in all_data]

    seed = random.randint(111111, 999999)
    info = create_folds(sentence_entities, folds, seed)

    for i in range(folds):
        train_indexs = info[f'fold-{i}']['train_indexs']
        test_indexs = info[f'fold-{i}']['test_indexs']

        os.makedirs(os.path.join(output_folder, f'fold-{i}'), exist_ok=True)

        with open(os.path.join(output_folder, f'fold-{i}', f'train.bio'), 'w') as f:
            for x in train_indexs:
                for token, label in zip(*all_data[x]):
                    f.write(f'{token}\t{label}\n')
                f.write('\n')
        with open(os.path.join(output_folder, f'fold-{i}', f'test.bio'), 'w') as f:
            for x in test_indexs:
                for token, label in zip(*all_data[x]):
                    f.write(f'{token}\t{label}\n')
                f.write('\n')

    with open(os.path.join(output_folder, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input files, separate with space, will concat them together
    parser.add_argument('--input_files', nargs='+', required=True)
    # output folder, will create per-fold folder in it
    parser.add_argument('--output_folder', required=True)
    # number of folds to make
    parser.add_argument('--folds', type=int, default=10)
    # label typing schema
    parser.add_argument('--schema', default="bio", choices=["bio", "iob", "iobes", "none"])
    args = parser.parse_args()
    print(vars(args))
    main(args.input_files, args.output_folder, args.folds, args.schema)
