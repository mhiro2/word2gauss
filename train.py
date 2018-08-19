import argparse
import sys
import time
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from model import GaussianEmbedding


class Corpus(Dataset):
    def __init__(self):
        self.word_index = defaultdict(lambda: len(self.word_index))

    @staticmethod
    def read_corpus(corpus_file):
        self = Corpus()
        counter = Counter()
        dataset = []

        with open(corpus_file, 'r', encoding='utf8') as f:
            for line in f:
                for word in line.split():
                    self.word_index[word]
                    counter[self.word_index[word]] += 1
                    dataset.append(self.word_index[word])

        self.index_word = {v: k for k, v in self.word_index.items()}
        self.counts = torch.LongTensor(
                          [counter[i] for i in range(len(counter))]
                      )
        self.dataset = torch.LongTensor(dataset)

        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class PairwiseWindowIter(object):
    def __init__(self, dataset, window, batch_size):
        self.current_position = 0
        self.dataset = dataset
        self.window = window
        self.batch_size = batch_size

        half_w = window % 2 + 1
        self.order = torch.randperm(len(dataset) - half_w * 2) + half_w
        self.offset = torch.cat((torch.arange(-half_w, 0),
                                 torch.arange(1, half_w + 1)))

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_position == -1:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size

        if i_end >= len(self.dataset):
            i_end = -1

        position = self.order[i:i_end].unsqueeze(1).repeat(1, self.window - 1)
        target = self.dataset[position]
        context = self.dataset[position + self.offset]

        self.current_position = i_end

        return target, context


def convert(batch, device):
    target, context = batch
    return target.to(device), context.to(device)


def train(model, dataset, args, device):
    model.train()
    optimizer = optim.Adam(model.parameters())
    start_time = time.time()

    for epoch in range(1, args.epoch + 1):
        train_iter = PairwiseWindowIter(dataset, args.window, args.batch_size)
        print('------------------------------')
        print('epoch: {}'.format(epoch))

        for i, batch in enumerate(train_iter):
            batch = convert(batch, device)
            loss = model(batch)

            elapsed = time.time() - start_time
            throuput = args.batch_size / elapsed
            prog = args.batch_size * (i + 1) / len(dataset) * 100
            print('\r  progress: {:.2f}% words/s: {:.2f}'.format(
                      min(prog, 100.), throuput
                  ), end='')
            sys.stdout.flush()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.embed.regularize_weights()

            start_time = time.time()

        print()
        print('  loss: {:.4f}'.format(loss.item()))

        if args.debug:
            torch.save(model.state_dict(),
                       '{}_epoch{}.pt'.format(args.train.replace('.', '_'),
                                              epoch)
                       )


def dump_result(model, index_word, args):
    model.to('cpu')
    mu_list, sigma_list = model.state_dict().values()

    with open(args.save, 'w') as f:
        f.write('{} {} {}\n'.format(len(index_word),
                                    args.size,
                                    args.covariance))

        for i, (mu, sigma) in enumerate(zip(mu_list, sigma_list)):
            mu_str = ' '.join('{0:.7f}'.format(i) for i in mu.tolist())
            sigma_str = ' '.join('{0:.7f}'.format(i) for i in sigma.tolist())
            f.write('{} {} {}\n'.format(index_word[i], mu_str, sigma_str))


def parse_args():
    size = 50
    window = 5
    epoch = 5
    batch_size = 128

    parser = argparse.ArgumentParser(description='Gaussian embedding')

    parser.add_argument('--train', type=str, required=True,
                        help='source corpus file')
    parser.add_argument('--save', type=str, required=True,
                        help='path to save the result model')
    parser.add_argument('--cuda', type=int, required=True,
                        help='''
                             set it to 1 for running on GPU, 0 for CPU
                             (GPU is 5x-10x slower than CPU)
                             ''')
    parser.add_argument('--epoch', '-e', default=epoch, metavar='N', type=int,
                        help='''
                             number of training epochs
                             (default: {})
                             '''.format(epoch))
    parser.add_argument('--size', '-s', default=size, metavar='N', type=int,
                        help='''
                             the dimension of embedding gaussian
                             (default: {})
                             '''.format(size))
    parser.add_argument('--batch_size', '-b', default=batch_size,
                        metavar='N', type=int,
                        help='''
                             minibatch size for training
                             (default:{})
                             '''.format(batch_size))
    parser.add_argument('--covariance', '-c', default='diagonal',
                        choices=['diagonal', 'spherical'],
                        help='''
                             covariance type ("diagonal", "spherical")
                             (default: diagonal)
                             ''')
    parser.add_argument('--window', '-w', default=window,
                        metavar='N', type=int,
                        help='window size (default: {})'.format(window))
    parser.add_argument('--seed', type=int, default='1234', help='random seed')
    parser.add_argument('--debug', '-d', action='store_true')

    args = parser.parse_args()

    try:
        if args.epoch < 1:
            raise ValueError('You must set --epoch >= 1')
        if args.size < 1:
            raise ValueError('You must set --size >= 1')
        if args.window < 1:
            raise ValueError('You must set --window >= 1')
        if args.batch_size < 1:
            raise ValueError('You must set --batch_size >= 1')
    except Exception as ex:
        parser.print_usage(file=sys.stderr)
        print(ex, file=sys.stderr)
        sys.exit()

    return args


def main():
    args = parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    torch.manual_seed(args.seed)

    dataset = Corpus.read_corpus(args.train)
    counts = dataset.counts

    print('vocab size: {}'.format(len(counts)))
    print('words in train file: {}'.format(len(dataset)))
    print()

    model = nn.Sequential()
    model.add_module('embed', GaussianEmbedding(args.size,
                                                counts,
                                                args.window,
                                                args.batch_size,
                                                args.covariance,
                                                device))
    model.to(device)
    print('Model summary:')
    print(model)
    print()

    train(model, dataset, args, device)
    dump_result(model, dataset.index_word, args)


if __name__ == '__main__':
    main()
