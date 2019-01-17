import math
import pandas as pd


def get_sample_weights(df, save_name=None, mu=1.0):
    """
    assign sample weights for each sample. rare sample have higher weights(linearly)
    refer to
    :param df:
    :param save_name:
    :param mu:
    :return:
    """
    label_list = df['Target'].tolist()
    import pickle
    import operator
    from functools import reduce
    from collections import Counter
    freq_count = dict(Counter(reduce(operator.add, map(lambda x: list(map(int, x.split(' '))), label_list))))
    total = sum(freq_count.values())
    keys = freq_count.keys()
    assert sorted(list(keys)) == list(range(len(keys)))
    class_weight = dict()
    class_weight_log = dict()
    for key in range(len(keys)):
        score = total / float(freq_count[key])
        score_log = math.log(mu * total / float(freq_count[key]))
        class_weight[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
        class_weight_log[key] = round(score_log, 2) if score_log > 1.0 else round(1.0, 2)

    rareness = [x[0] for x in sorted(freq_count.items(), key=operator.itemgetter(1))]

    weights = []
    sample_labels = list(map(lambda x: list(map(int, x.split(' '))), label_list))
    for labels in sample_labels:
        for rare_label in rareness:
            if rare_label in labels:
                weights.append(class_weight[rare_label])
                break

    assert len(weights) == len(label_list)
    return weights
    # with open(save_name, 'wb') as f:
    #     pickle.dump(weights, f)
    # print("%d weights saved into %s" % (len(label_list), save_name))
