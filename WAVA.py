import numpy as np
import matplotlib.pyplot as plt
import commpy.modulation as cm
import commpy.channelcoding.convcode as cc
import commpy.channels as cch
from numpy import array
import tb_encoder
import tb_viterbi
from multiprocessing.pool import ThreadPool


def wava(upperbound, snr, trellis):

    BPSK_Mode = cm.PSKModem(2)
    last_state1 = 0
    last_metric1 = 0
    last_state2 = 0
    last_metric2 = 0
    BitErrors = 0

    for i in range(upperbound):
        message_bits = np.random.randint(0, 2, 128)
        # tail biting encode
        tbencodedbits = tb_encoder.conv_encode_tb(message_bits,trellis)
        # BPSK
        BPSK_bits = BPSK_Mode.modulate(tbencodedbits)
        # through the channel
        receive_code_channel = cch.awgn(BPSK_bits,snr)
        # demodulate
        receive_code = BPSK_Mode.demodulate(receive_code_channel, demod_type='hard')
        # no tail viterbi
        decoded_bits_notb = tb_encoder.viterbi_decoder_notail(receive_code, trellis, last_state1, last_metric1)

        decoded_bits_tb = tb_viterbi.tb_viterbi_decode(receive_code, trellis, last_state2, last_metric2)

        # decoded_bits_notb = async_result1.get()
        # decoded_bits_tb = async_result2.get()

        # reset the values of last state
        last_state1 = decoded_bits_notb[1]
        last_metric1 = decoded_bits_notb[2]
        last_state2 = decoded_bits_tb[1]
        last_metric2 = decoded_bits_tb[2]
        print(i)
        # compare these two results
        if np.array_equal(decoded_bits_tb[0], decoded_bits_notb[0]):
            result1 = decoded_bits_notb[0]
        elif i == upperbound - 1:
            result1 = decoded_bits_tb[0]
        else:
            continue
        BitErrors = str(array(message_bits) ^ result1).count('1')
        print(BitErrors)
        break
    return BitErrors


def wavarunner(message_numbers, upperbound, snr, index):
    memory = array([8])
    g_matrix = array([[0o515, 0o677]])
    trellis = cc.Trellis(memory, g_matrix)
    total_errors = 0
    for i in range(message_numbers):
        total_errors += wava(upperbound, snr, trellis)
        print("pass: {}, {}".format(i, index))
    return total_errors/(128*message_numbers)


# x = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
# x = [1, 2, 3, 4, 5, 6]
x = [0,0.5,1,1.5,2,2.5,3]
y = []
index = 0
for i in x:
    result = wavarunner(5, 3, i, index)
    index += 1
    y.append(result)

plt.plot(x, y)
plt.grid(True)
plt.title('Performance of Wrap-around Viterbi Algorithm')
plt.xlabel('Eb/N0 in dB')
plt.yscale('log')
# plt.ylim(10**(-5),10**(0))
plt.ylabel('BER')
plt.show()
