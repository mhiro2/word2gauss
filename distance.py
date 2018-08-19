import sys
import numpy as np


NUM_RESULT = 10


def main():
    with open(sys.argv[1], 'r') as f:
        info, covariance = f.readline().rsplit(maxsplit=1)
        num_vocab, embed_size = [int(x) for x in info.split()]

        w_mu = np.empty((num_vocab, embed_size), dtype=np.float32)
        # if covariance == 'diagonal':
        #     w_sigma = np.empty((num_vocab, embed_size), dtype=np.float32)
        # elif covariance == 'spherical':
        #     w_sigma = np.empty((num_vocab, 1), dtype=np.float32)

        index_word = {}
        word_index = {}

        for i, line in enumerate(f):
            ls = line.split()
            word = ls[0]
            index_word[i] = word
            word_index[word] = i
            w_mu[i] = np.array([float(s) for s in ls[1:embed_size + 1]],
                               dtype=np.float32)
            # w_sigma[i] = np.array([float(s) for s in ls[embed_size + 1:]],
            #                       dtype=np.float32)

    mu_s = np.sqrt((w_mu * w_mu).sum(1))
    w_mu_norm = w_mu / mu_s.reshape((mu_s.shape[0], 1))

    try:
        while True:
            w = input('>> ')
            if w not in word_index:
                print('{} is not found.'.format(w))
                continue

            m = w_mu_norm[word_index[w]]

            cosine_sim = w_mu.dot(m)

            print('query: {}'.format(w))
            print('--------------------')
            count = 0

            for i in (-cosine_sim).argsort():
                if np.isnan(cosine_sim[i]):
                    continue
                if index_word[i] == w:
                    continue
                print('{}: {}'.format(index_word[i], cosine_sim[i]))
                count += 1

                if count == NUM_RESULT:
                    break

    except EOFError:
        pass

    except KeyboardInterrupt:
        sys.exit()


if __name__ == '__main__':
    main()
