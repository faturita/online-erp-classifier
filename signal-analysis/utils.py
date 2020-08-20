import numpy as np
import mne
import operator

def flatten_experiences(experiences):
    """
    Given a list of epochs data list and a list of label lists,
    it returns a list containing all epochs data and another one containing all labels.
    :param experiences: list of list of epochs with labels
    :return: a list containing all epochs data and another one containing all labels
    """

    samples = [sample for experience in experiences for sample in experience]
    epochs_data = [epoch for epoch,_ in samples]
    labels = [label for _, label in samples]
    return epochs_data, labels


def grid_lights_rewards(prediction):
    if prediction is None:
        return 10
    else:
        return -1 * prediction


def prediction_2_reward(predictions, reward_function):
    return reward_function(predictions)


def state_reward_merge(state_files, rewards, reward_function, output_file):
    rewards = list(rewards)
    with open(output_file, "w") as out:
        experiment_name = None
        file_content = ""
        for state_file in state_files:
            with open(state_file, "r") as inp:
                if experiment_name is None:
                    experiment_name = inp.readline()
                    file_content += experiment_name
                elif experiment_name != inp.readline():
                    raise Exception("Experiments are different.")

                is_first_state = True

                for line in inp:
                    if line != "\n":
                        file_content += line
                    else:
                        if is_first_state:
                            is_first_state = False
                            file_content += str(reward_function(0)) + "\n"
                        else:
                            file_content += str(rewards.pop(0)) + "\n"
                file_content += "\n" + str(reward_function(None)) + "\n--\n"

        out.write(file_content)


def merge_predictions_with_states(predictions, state_files, reward_function, output_file):
    rewards = prediction_2_reward(predictions, reward_function)
    state_reward_merge(state_files, rewards, reward_function, output_file)

def get_separability_values(file, evoked_hits, evoked_no_hits, index):
    aux = file.split("/")
    name = aux[2] + " - " + aux[3]
    txt_file = file[:-5] + "-" + str(index) + ".txt"
    max_dif_channel = 0
    max_channel = 0
    f = open(txt_file, "w")

    for channel in range(len(evoked_hits.data)):
        aux = evoked_hits.data[channel] - evoked_no_hits.data[channel]

        difference = np.sum(np.abs(aux))
        if max_dif_channel < difference:
            max_dif_channel = difference
            max_channel = channel
        f.write(str(difference) + "\n")
    f.close()
    f1 = open("../notepads/maximos.txt", "a")
    f1.write(name + " - game[" + str(index) + "] - channel[" + str(max_channel) + "]:" + str(max_dif_channel) + "\n")
    f1.close()

def plot_hits_and_no_hits(evoked_hits, evoked_no_hits):
    edi = {'Hits': evoked_hits, 'No Hits': evoked_no_hits}
    mne.viz.plot_compare_evokeds(edi, picks=['FZ'])
    mne.viz.plot_compare_evokeds(edi, picks=['CZ'])
    mne.viz.plot_compare_evokeds(edi, picks=['P3'])
    mne.viz.plot_compare_evokeds(edi, picks=['PZ'])
    mne.viz.plot_compare_evokeds(edi, picks=['P4'])
    mne.viz.plot_compare_evokeds(edi, picks=['PO7'])
    mne.viz.plot_compare_evokeds(edi, picks=['PO8'])
    mne.viz.plot_compare_evokeds(edi, picks=['OZ'])


def order(file):
    f = open(file, "r")
    mydict = dict()
    for line in f:
        aux = line.split(":")
        mydict.update({aux[0]: float(aux[1])})

    # mydict = collections.OrderedDict(sorted(mydict.items()))

    sorted_x = sorted(mydict.items(), key=operator.itemgetter(1), reverse=True)

    # for k, v in mydict.items(): print(k, v)
    for a in sorted_x:
        print(a)

def group_and_average_hits_and_no_hits(epochs_data, labels, n):

    if n > len(epochs_data) or n == 1:
        return epochs_data, labels
    new_epochs_data = []
    new_labels = []
    sub_index_hits = 0
    sub_index_no_hits = 0
    sum_n_hits = np.zeros(shape=(8,21))
    sum_n_no_hits = np.zeros(shape=(8,21))
    length = len(epochs_data)
    for index in range(length):

        if labels[index] == 1:
            sum_n_hits = sum_n_hits + epochs_data[index]
            sub_index_hits = sub_index_hits + 1
        elif labels[index] == 0:
            sum_n_no_hits = sum_n_no_hits + epochs_data[index]
            sub_index_no_hits = sub_index_no_hits + 1

        if sub_index_hits == n:
            new_epochs_data.append(sum_n_hits/n)
            new_labels.append(1)
            sub_index_hits = 0
            sum_n_hits = np.zeros(shape=(8,21))

        if sub_index_no_hits == n:
            new_epochs_data.append(sum_n_no_hits/n)
            new_labels.append(0)
            sub_index_no_hits = 0
            sum_n_no_hits = np.zeros(shape=(8,21))

    return new_epochs_data, new_labels