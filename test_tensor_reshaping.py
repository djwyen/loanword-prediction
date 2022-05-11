import unittest

import torch
import numpy as np

BATCH_SIZE = 64
NUM_PANPHON_FEATURES = 24
HIDDEN_DIM = 32
MAX_SEQ_LEN_NO_STOP = 20
MAX_SEQ_LEN_WITH_STOP = MAX_SEQ_LEN_NO_STOP + 1 # due to the added stop token at the end

class TestTensorReshaping(unittest.TestCase):
    def setUp(self):
        # arbitrary, but I happened to choose realistic values
        self.batch_size = 32
        self.max_seq_len = 21
        self.n_features = 22
        self.hidden_size = 32
        self.n_categories = 3 # the number of categories each feature can take on; for us, this is {+,-,0}
        # of the encoder we are presumably modeling the behavior of
        self.enc_layers = 3
        self.enc_directions = 2

    def test_multilayer_stacking(self):
        range_size = self.enc_layers * self.batch_size * 2 * self.hidden_size
        x = list(range(range_size))
        x = torch.tensor(np.array(x))
        x = torch.reshape(x, (self.enc_layers, self.batch_size, 2*self.hidden_size)) # (n_layers, N, 2*H_out)
        # check that the reshapes work as expected
        self.assertEqual(x.shape, (self.enc_layers, self.batch_size, 2*self.hidden_size))
        self.assertEqual(x.numpy()[-1, -1, -1], range_size - 1)
        # I'm lazy, so for now we look at these by manual inspection of each slice
        print(x.numpy()[-1,-1,:])

        x = torch.permute(x, (1,0,2)) # (N, n_layers, 2*H_out)
        self.assertEqual(x.shape, (self.batch_size, self.enc_layers, 2*self.hidden_size))
        self.assertEqual(x.numpy()[-1, -1, -1], range_size - 1)
        print(x.numpy()[-1,-1,:])
        x = torch.reshape(x, (x.size(0), -1)) # (N, 2*n_layers*H_out)
        self.assertEqual(x.shape, (self.batch_size, 2*self.enc_layers*self.hidden_size))
        self.assertEqual(x.numpy()[-1, -1], range_size - 1)
        print(x.numpy()[-1,:])


    def test_stacking_hidden_with_onehots(self):
        x = range(self.batch_size * self.max_seq_len * self.n_features * self.n_categories)
        x = torch.tensor(np.array(x))
        x = torch.reshape(x, (self.batch_size, self.max_seq_len, self.n_features, self.n_categories)) # x: (N, L, H_in, 3)
        print(x.numpy()[-1,-1,0,:])
        print(x.numpy()[-1,-1,1,:])
        print(x.numpy()[-1,-1,-1,:])
        # we want to show that after stacking the fourth dimension for a given slice (here the (-1,-1)th 2D tensor)
        # the order is preserved. That means we want to see the value ranging from the first value in the
        # first 3-vector to the last value of the last 3-vector, monotonically
        start_of_slice = x.numpy()[-1,-1,0,0]
        end_of_slice = x.numpy()[-1,-1,-1,2]
        x = torch.reshape(x, (self.batch_size, self.max_seq_len, -1)) # x: (N, L, 3*H_in)
        stacked_slice = x.numpy()[-1,-1,:]
        self.assertEqual(stacked_slice[0], start_of_slice)
        self.assertEqual(stacked_slice[-1], end_of_slice)
        self.assertEqual(len(stacked_slice), self.n_categories * self.n_features)
        # assert that the slice is monotonic, which we could also confirm by manual inspection:
        self.assertTrue(all([x < y for x, y in zip(stacked_slice, stacked_slice[1:])]))
        print(stacked_slice)


if __name__ == '__main__':
    unittest.main()