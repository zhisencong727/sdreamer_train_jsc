from scipy.io import loadmat,savemat
import numpy as np

filename = "/Users/jsc727/Downloads/830.mat"
matfile = loadmat(filename)
emg = matfile['emg'].flatten()
emg_original = emg
emg = np.abs(emg)
padding_size = 64 * 512
padding_arr = np.zeros(padding_size)
emg_padded = np.concatenate((padding_arr, emg))

window_size = padding_size

cumsum_emg_padded = np.concatenate([[0], np.cumsum(emg_padded)])

indices = np.arange(window_size, len(emg_padded))
window_sums = cumsum_emg_padded[indices + 1] - cumsum_emg_padded[indices - window_size + 1]
emg_moving_avg = window_sums / window_size

print("EMG.shape",emg.shape)
print("EMG_MOVING_AVG.size",emg_moving_avg.shape)


emg_minus_moving_avg = emg_original/emg_moving_avg

matfile['emg_moving_avg'] = emg_moving_avg
matfile['emg_minus_moving_avg'] = emg_minus_moving_avg
savemat(filename,matfile)