import numpy as np
import matplotlib.pyplot as plt
import commpy.modulation as cm
import operator
from commpy.utilities import dec2bitarray, bitarray2dec, hamming_dist, euclid_dist
import commpy.channelcoding.convcode as cc
import commpy.channels as cch
from numpy import array
import tb_encoder
import shortviterbi1
import math
import scipy.stats


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


def stack_decoder(receive_code, metrics, trellis, initial_state, initial_metric, discarded_path='2'):
    k = trellis.k
    n = trellis.n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs
    next_state_table = trellis.next_state_table
    output_table = trellis.output_table
    r_code = convert(receive_code)
    final_state = 0
    path_metric = {discarded_path: -10**8}
    list_length = 128
    M = trellis.total_memory

    for i in range(10**6):  # a large number to guarantee the sufficient iterations times of stack decoder
        # each receiving code to be decoded
        if i == 0:
            for j in range(number_inputs):
                p_metric = 0.0
                output = output_table[initial_state][j]
                output_array = dec2bitarray(output, n)
                r_code_array = array(r_code[2 * i:(2 * i + 2)])
                p_metric += (soft_metric(metrics, output_array[0], r_code_array[0]) + soft_metric(metrics, output_array[1], r_code_array[1]))
                path_metric[str(j)] = p_metric + initial_metric
            path_metric = dict(sorted(path_metric.items(), key=operator.itemgetter(1)))
            # dictionary always go in the order as from small to big

        elif i == 1:
            # get the path with best metric right now
            # current_best_path = dict(sorted(path_metric.items(),key = operator.itemgetter(1))[len(path_metric)-1])
            # last_input = max(path_metric, key=lambda k: path_metric[k])
            last_input = list(path_metric)[len(path_metric) - 1]
            current_state = next_state_table[initial_state][int(last_input)]
            for j in range(number_inputs):
                p_metric = 0.0
                output = output_table[current_state][j]
                output_array = dec2bitarray(output, n)
                r_code_array = array(r_code[2 * i:(2 * i + 2)])
                p_metric += (soft_metric(metrics, output_array[0], r_code_array[0]) + soft_metric(metrics, output_array[1], r_code_array[1]))

                path_metric[str(last_input) + str(j)] = path_metric[last_input] + p_metric

            path_metric.pop(str(last_input))
            path_metric = dict(sorted(path_metric.items(), key=operator.itemgetter(1)))
            #  dictionary always go in the order as from small to big

        elif i >= 2:
            best_path_sofar = list(path_metric)[len(path_metric) - 1]  # ????? it's an int other than string
            if len(best_path_sofar) >= 2:
                current_state = initial_state
                for m in str(best_path_sofar):
                    # current_state = int(str(best_path_sofar)[len(best_path_sofar) - 1] + str(best_path_sofar)
                    # [len(best_path_sofar) - 2])
                    current_state = next_state_table[current_state][int(m)]
            elif len(best_path_sofar) == 1:
                current_state = next_state_table[initial_state][int(best_path_sofar)]
            else:
                current_state = initial_state
            for j in range(number_inputs):
                p_metric = 0.0
                output = output_table[current_state][j]
                output_array = dec2bitarray(output, n)
                r_code_array = array(r_code[2 * len(best_path_sofar):(2 * len(best_path_sofar) + 2)])
                p_metric += (soft_metric(metrics, output_array[0], r_code_array[0]) + soft_metric(metrics, output_array[1], r_code_array[1]))

                path_metric[str(best_path_sofar) + str(j)] = path_metric[best_path_sofar] + p_metric

            path_metric.pop(str(best_path_sofar))
            path_metric = dict(sorted(path_metric.items(), key=operator.itemgetter(1)))
            # dictionary always go in the order from small to big
            if len(path_metric) > list_length:
                path_metric.pop(list(path_metric)[0])
            if len(list(path_metric)[len(path_metric) - 1]) >= len(r_code)*0.5:
                break

    decoded_bits = list(path_metric)[len(path_metric) - 1]
    ini_state_bit = list(dec2bitarray(initial_state, 8)[::-1])
    final_metric = path_metric[decoded_bits]
    decoded_code = [0]*len(decoded_bits)
    for k in range(len(decoded_bits)):
        decoded_code[k] = int(decoded_bits[k])
    # decoded_code += ini_state_bit
    final_state = bitarray2dec(decoded_code[(int(len(r_code) * 0.5) - M):][::-1])

    return decoded_code, final_metric, final_state, decoded_bits


def stack_decoder_fixed_tb(receive_code, metrics, trellis, initial_state, initial_metric, discarded_path='2'):
    k = trellis.k
    n = trellis.n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs
    next_state_table = trellis.next_state_table
    output_table = trellis.output_table
    r_code = convert(receive_code)
    final_state = 0
    if len(discarded_path) > 1:
        path_metric = {discarded_path[120:]: -10**8}
    else:
        path_metric = {discarded_path: -10 ** 8}
    list_length = 128
    M = trellis.total_memory
    initial_state_bin = dec2bitarray(initial_state, M)[::-1]

    for i in range(10**6):  # a large number to guarantee the sufficient iterations times of stack decoder
        # each receiving code to be decoded
        if i == 0:
            for j in range(number_inputs):
                p_metric = 0.0
                output = output_table[initial_state][j]
                output_array = dec2bitarray(output, n)
                r_code_array = array(r_code[2 * i:(2 * i + 2)])
                p_metric += (soft_metric(metrics, output_array[0], r_code_array[0]) + soft_metric(metrics, output_array[1], r_code_array[1]))
                path_metric[str(j)] = p_metric + initial_metric
            path_metric = dict(sorted(path_metric.items(), key=operator.itemgetter(1)))
            # dictionary always go in the order as from small to big

        elif i == 1:
            # get the path with best metric right now
            # current_best_path = dict(sorted(path_metric.items(),key = operator.itemgetter(1))[len(path_metric)-1])
            # last_input = max(path_metric, key=lambda k: path_metric[k])
            last_input = list(path_metric)[len(path_metric) - 1]
            current_state = next_state_table[initial_state][int(last_input)]
            for j in range(number_inputs):
                p_metric = 0.0
                output = output_table[current_state][j]
                output_array = dec2bitarray(output, n)
                r_code_array = array(r_code[2 * i:(2 * i + 2)])
                p_metric += (soft_metric(metrics, output_array[0], r_code_array[0]) + soft_metric(metrics, output_array[1], r_code_array[1]))

                path_metric[str(last_input) + str(j)] = path_metric[last_input] + p_metric

            path_metric.pop(str(last_input))
            path_metric = dict(sorted(path_metric.items(), key=operator.itemgetter(1)))
            #  dictionary always go in the order as from small to big

        elif i >= 2:
            best_path_sofar = list(path_metric)[len(path_metric) - 1]  # ????? it's an int other than string
            if len(best_path_sofar) >= 2:
                current_state = initial_state
                for m in str(best_path_sofar):
                    # current_state = int(str(best_path_sofar)[len(best_path_sofar) - 1] + str(best_path_sofar)
                    # [len(best_path_sofar) - 2])
                    current_state = next_state_table[current_state][int(m)]
            elif len(best_path_sofar) == 1:
                current_state = next_state_table[initial_state][int(best_path_sofar)]
            else:
                current_state = initial_state

            if len(best_path_sofar) < len(receive_code) * 0.5 - M:  # to manage tail-biting feature
                for j in range(number_inputs):
                    p_metric = 0.0
                    output = output_table[current_state][j]
                    output_array = dec2bitarray(output, n)
                    r_code_array = array(r_code[2 * len(best_path_sofar):(2 * len(best_path_sofar) + 2)])
                    p_metric += (soft_metric(metrics, output_array[0], r_code_array[0]) + soft_metric(metrics, output_array[1], r_code_array[1]))

                    path_metric[str(best_path_sofar) + str(j)] = path_metric[best_path_sofar] + p_metric

                path_metric.pop(str(best_path_sofar))
                path_metric = dict(sorted(path_metric.items(), key=operator.itemgetter(1)))
            else:
                for j in range(number_inputs):
                    if j == initial_state_bin[len(best_path_sofar) - int((len(receive_code) * 0.5) - M)]:
                        p_metric = 0.0
                        output = output_table[current_state][j]
                        output_array = dec2bitarray(output, n)
                        r_code_array = array(r_code[2 * len(best_path_sofar):(2 * len(best_path_sofar) + 2)])
                        p_metric += (soft_metric(metrics, output_array[0], r_code_array[0]) + soft_metric(metrics, output_array[1], r_code_array[1]))

                        path_metric[str(best_path_sofar) + str(j)] = path_metric[best_path_sofar] + p_metric
                    else:
                        path_metric[str(best_path_sofar) + str(j)] = -10**9

                path_metric.pop(str(best_path_sofar))
                path_metric = dict(sorted(path_metric.items(), key=operator.itemgetter(1)))
            # dictionary always go in the order from small to big
            if len(path_metric) > list_length:
                path_metric.pop(list(path_metric)[0])
            if len(list(path_metric)[len(path_metric) - 1]) >= len(r_code)*0.5:
                break

    decoded_bits = list(path_metric)[len(path_metric) - 1]
    # ini_state_bit = list(dec2bitarray(initial_state, 8)[::-1])
    final_metric = path_metric[decoded_bits]
    decoded_code = [0]*len(decoded_bits)
    for k in range(len(decoded_bits)):
        decoded_code[k] = int(decoded_bits[k])
    # decoded_code += ini_state_bit
    final_state = bitarray2dec(decoded_code[(int(len(r_code) * 0.5) - M):][::-1])

    return decoded_code, final_metric, final_state


def newmethodrunner(iterationtimes, eb_n0, index, metrics):
    memory = array([8])
    g_matrix = array([[0o515, 0o677]])
    trellis = cc.Trellis(memory, g_matrix)
    BPSKMod = cm.PSKModem(2)
    # set the message size
    total_error = 0
    cer = 0

    for d in range(iterationtimes):
        message_bits = np.random.randint(0, 2, 128)
        result = 0
        # print(message_bits)
        encoded_code = tb_encoder.conv_encode_tb(message_bits, trellis)
        BPSK_modedbits = BPSKMod.modulate(encoded_code)
        r_code = cch.awgn(BPSK_modedbits, eb_n0 + 3, 0.5)
        # r_code = BPSKMod.demodulate(AWGNreceived_bits, demod_type='hard')
        # part_code = BPSKMod.demodulate(r_code[240:], demod_type='hard')
        part_decode = shortviterbi1.short_viterbi(np.append(r_code[240:], r_code[0:64]), trellis, 'unquantized')[0]
        # part_decode = message_bits[120:]
        # print(part_decode)
        # print(len(part_decode))
        initial_code = part_decode[0:8][::-1]
        # initial_code = message_bits[120:][::-1]
        initial_state = bitarray2dec(initial_code)
        # print(initial_state)
        initial_metric = 0
        result = stack_decoder_fixed_tb(r_code, metrics, trellis, initial_state, initial_metric)[0]

        print("pass: {}, {}".format(d, index))
        # print(result)
        bit_error = str(message_bits ^ array(result)).count('1')
        if bit_error > 0:
            cer += 1
        # print(bit_error)
        total_error += bit_error

    averageber = total_error/(iterationtimes * 128)
    a_cer = cer/iterationtimes
    return a_cer, averageber


# x = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]
x = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]
y1 = [] # cer
y2 = [] # ber
# z = [2, 2.5, 3, 3.5, 4, 4.5, 5]
z = [2]
index = 0

for f in z:
    pyx = pyx_matrix(f)
    metrics = fano_metrics(pyx)
    aver_ber = newmethodrunner(200, f, index, metrics)
    print(aver_ber)
    y1.append(aver_ber[0])
    y2.append(aver_ber[1])
    index += 1
    # time.sleep(60)

plt.plot(z, y1)
plt.plot(z,y2)
plt.grid(True)
plt.title('Performance of Soft Stack Algorithm(with VA)')
plt.xlabel('Eb/N0 in dB')
plt.yscale('log')
# plt.ylim(10**(-5),10**(0))
plt.ylabel('CER')
plt.show()