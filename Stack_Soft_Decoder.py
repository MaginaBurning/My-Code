import numpy as np
import matplotlib.pyplot as plt
import math
import commpy.modulation as cm
import operator
from commpy.utilities import dec2bitarray
import commpy.channelcoding.convcode as cc
import commpy.channels as cch
from numpy import array
import scipy.stats
import time


def convert(receive_code):
    # sig = math.sqrt(0.5 / 10 ** (snr / 10))
    y = [0] * len(receive_code)

    for i in range(len(receive_code)):
        if receive_code[i].real >= 1:
            y[i] = '14'
        elif receive_code[i].real >= 2/3 and receive_code[i].real < 1 :
            y[i] = '13'
        elif receive_code[i].real >= 1/3 and receive_code[i].real < 2/3:
            y[i] = '12'
        elif receive_code[i].real >= 0 and receive_code[i].real < 1/3:
            y[i] = '11'
        elif receive_code[i].real >= -1/3 and receive_code[i].real < 0:
            y[i] = '01'
        elif receive_code[i].real >= -2/3 and receive_code[i].real < -1/3:
            y[i] = '02'
        elif receive_code[i].real >= -1 and receive_code[i].real < -2/3:
            y[i] = '03'
        elif receive_code[i].real < -1:
            y[i] = '04'

    return y


def pyx_matrix(snr):
    u = 1
    sig = 10**(-snr/20)
    pyx = np.zeros((3, 4))
    pyx[0, :] = [1, 2 / 3, 1 / 3, 0]
    pyx[1, 0] = 1 - scipy.stats.norm(u, sig).cdf(1)
    pyx[2, 0] = scipy.stats.norm(u, sig).cdf(-1)
    for k in range(1, 4):
        pyx[1, k] = scipy.stats.norm(u, sig).cdf(pyx[0, k - 1]) - scipy.stats.norm(u, sig).cdf(pyx[0, k])
        pyx[2, k] = -scipy.stats.norm(u, sig).cdf(-pyx[0, k - 1]) + scipy.stats.norm(u, sig).cdf(-pyx[0, k])

    pyx[1, :] = pyx[1, :][::-1]
    pyx[2, :] = pyx[2, :][::-1]
    return pyx


def fano_metrics(pyx):
    metrics = np.zeros((2, 4))
    for b in range(4):
        metrics[0, b] = math.log(2 * pyx[1, b] / (pyx[1, b] + pyx[2, b]), 2) - 0.5
        metrics[1, b] = math.log(2 * pyx[2, b] / (pyx[1, b] + pyx[2, b]), 2) - 0.5
    return metrics


def soft_metric(metrics, state, r_code_cpnt):

    if state == int(r_code_cpnt[0]):
        return round(metrics[0, int(r_code_cpnt[1])-1]*20, 1)
    elif state != int(r_code_cpnt[0]):
        return round(metrics[1, int(r_code_cpnt[1])-1] * 20, 1)


def stack_soft_decoder(receive_code, metrics, trellis):
    # k = trellis.k
    n = trellis.n
    # number_states = trellis.number_states
    number_inputs = trellis.number_inputs
    next_state_table = trellis.next_state_table
    output_table = trellis.output_table
    path_metric = {}
    r_code = convert(receive_code)
    list_length = 128
    init_time = time.time()

    for i in range(10 ** 8):
        # a long sequence to guarantee the iterations times of stack decoder
        # each receiving code to be decoded

        if i == 0:
            for j in range(number_inputs):
                p_metric = 0.0
                output = output_table[0][j]
                output_array = dec2bitarray(output, n)
                r_code_array = array(r_code[2 * i:(2 * i + 2)])
                p_metric += soft_metric(metrics, output_array[0], r_code_array[0]) + soft_metric(metrics, output_array[1], r_code_array[1])
                path_metric[str(j)] = p_metric
            path_metric = dict(sorted(path_metric.items(), key=operator.itemgetter(1)))
            # dictionary always go in the order from small to big

        elif i == 1:
            # get the path with best metric right now
            # current_best_path = dict(sorted(path_metric.items(),key = operator.itemgetter(1))[len(path_metric)-1])
            last_input = list(path_metric)[len(path_metric) - 1]
            # last_input = max(path_metric, key=lambda k: path_metric[k])
            current_state = next_state_table[0][int(last_input)]
            for j in range(number_inputs):
                p_metric = 0.0
                output = output_table[current_state][j]
                output_array = dec2bitarray(output, n)
                r_code_array = array(r_code[2 * i:(2 * i + 2)])
                p_metric += soft_metric(metrics, output_array[0], r_code_array[0]) + soft_metric(metrics, output_array[1], r_code_array[1])
                path_metric[str(last_input) + str(j)] = path_metric[last_input] + p_metric

            path_metric.pop(str(last_input))
            path_metric = dict(sorted(path_metric.items(), key=operator.itemgetter(1)))
            # dictionary always go in the order as from small to big

        elif i >= 2:
            best_path_sofar = list(path_metric)[len(path_metric) - 1]  # ????it's an int other than string
            # best_path_sofar = max(path_metric, key=lambda k: path_metric[k])
            if len(best_path_sofar) >= 2:
                current_state = 0
                for m in best_path_sofar:
                    # current_state = int(str(best_path_sofar)[len(best_path_sofar) - 1] + str(best_path_sofar)
                    # [len(best_path_sofar) - 2])
                    current_state = next_state_table[current_state][int(m)]
            elif len(best_path_sofar) == 1:
                current_state = next_state_table[0][int(best_path_sofar)]
            else:
                current_state = 0
            for j in range(number_inputs):
                p_metric = 0.0
                output = output_table[current_state][j]
                output_array = dec2bitarray(output, n)
                r_code_array = array(r_code[2 * len(best_path_sofar):(2 * len(best_path_sofar) + 2)])
                p_metric += soft_metric(metrics, output_array[0], r_code_array[0]) + soft_metric(metrics, output_array[1], r_code_array[1])
                path_metric[str(best_path_sofar) + str(j)] = path_metric[best_path_sofar] + p_metric

            path_metric.pop(str(best_path_sofar))
            path_metric = dict(sorted(path_metric.items(), key=operator.itemgetter(1)))
            # dictionary always go in the order as from small to big
            if len(path_metric) > list_length:
                path_metric.pop(list(path_metric)[0])
            if len(list(path_metric)[len(path_metric) - 1]) >= math.ceil(0.5 * len(r_code)):
                break
            if (time.time() - init_time) > 2:
                break

    decoded_bits = max(path_metric, key=lambda k: path_metric[k])
    decoded_code = [0] * len(decoded_bits)
    for k in range(len(decoded_bits)):
        decoded_code[k] = int(decoded_bits[k])

    return decoded_code


def stack_runner(IterationTimes, snr, index, metrics):
    memory = array([8])
    g_matrix = array([[0o515, 0o677]])
    trellis = cc.Trellis(memory, g_matrix)
    BPSKMod = cm.PSKModem(2)
    CodeError = 0

    total_result = 0.0
    for i in range(IterationTimes):
        # encode

        # set the message size
        message_bits = np.random.randint(0, 2, 128)
        # print(message_bits)
        encoded_code = cc.conv_encode(message_bits, trellis)
        BPSK_modedbits = BPSKMod.modulate(encoded_code)
        AWGNreceived_bits = cch.awgn(BPSK_modedbits, snr+3, 0.5)
        result = stack_soft_decoder(AWGNreceived_bits, metrics, trellis)

        print("pass: {}, {}".format(i, index))
        if len(result) != 136:
            continue
        else:
            BitErrors = str(array(message_bits) ^ array(result[0:len(message_bits)])).count('1')

            # print(result)
            # print(BitErrors)
            # print(result_int)
            # print(message_bits_int)
            BER = BitErrors / 128
            if BitErrors > 0:
                CodeError +=1
            total_result += BER
            # i += 1
            # print(i)

            # print(result)
    CER = CodeError / IterationTimes
    AverageBER = total_result / IterationTimes
    return CER


# print(stack_runner(10,2))

y1 = []
index = 0
x = [1,1.5,2,2.5,3,3.5,4]
# x = [3.5,4,4.5,5,5.5,6]
# x = [0, 1, 2, 3, 4, 5, 6]
for i in x:
    pyx = pyx_matrix(i)
    metrics = fano_metrics(pyx)
    aver_ber = stack_runner(100, i, index, metrics)
    print(aver_ber)
    y1.append(aver_ber)
    index += 1

plt.plot(x, y1)
plt.grid()
plt.title('Performance of Stack Algorithm-soft decision')
plt.xlabel('Eb/N0 in dB')
plt.yscale('log')
# plt.ylim(10**(-5),10**(0))
plt.ylabel('CER')
plt.show()
