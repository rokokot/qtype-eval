# -*- coding: utf-8  -*-
# !/usr/bin/python

import operator


def dict_counter(dictionary, key):
    """

    :param dictionary:
    :param key:
    :return: (int)
    """
    try:
        dictionary[key] += 1
    except KeyError:
        dictionary[key] = 1


def dict_dump_features(dictionary, prefix):
    try:
      if prefix == '':
        return sorted({str(key): float(value)
                       for (key, value) in dictionary.items()}.items(),
                      key=operator.itemgetter(1), reverse=True)

      return sorted({prefix + '_' + str(key): float(value)
                    for (key, value) in dictionary.items()}.items(),
                    key=operator.itemgetter(1), reverse=True)
    except KeyError:
        print('KeyError')
        return None


def dict_distribution(dictionary, prefix, multiplier_value=1):
    try:
        if prefix == '':
            return sorted({str(key): (multiplier_value * float(value)) / sum(dictionary.values())
                           for (key, value) in dictionary.items()}.items(),
                          key=operator.itemgetter(1), reverse=True)

        return sorted({prefix + '_' + str(key): (multiplier_value * float(value)) / sum(dictionary.values())
                       for (key, value) in dictionary.items()}.items(),
                      key=operator.itemgetter(1), reverse=True)
    except KeyError:
        print('KeyError')
        return None


def get_from_dict(dictionary, key):
    try:
        return dictionary[key]
    except KeyError:
        return 0


def ratio(dividend, divisor, multplier_value=1):
    try:
        return (multplier_value * dividend) / float(divisor)
    except ZeroDivisionError:
        return 0  # Added return 0 (before --> return, it returned None! Not right)


def vectorize(documents):
    feats = {}
    # Print header START
    for key, features in documents.items():
        for feature, value in features.items():
            if isinstance(value, float) or isinstance(value, int):
                feats[feature] = None
            else:
                try:
                    feats[feature] = list(set(feats[feature] + [i[0] for i in value]))
                except KeyError:
                    feats[feature] = [i[0] for i in value]

    s = 'identifier' + '\t'
    for feat, val in feats.items():
        if val:
            # print(val)
            for i in val:
                s += str(i) + '\t'
        else:
            s += feat + '\t'
    to_print = (s[:-1]) + '\n'
    # PRINT HEADER END

    for identifier, document in documents.items():
        s = identifier + '\t'
        for feat, val in feats.items():
            if val:
                for i in val:
                    try:
                        s += str(dict(document[feat])[i]) + '\t'
                    except KeyError:
                        s += '0' + '\t'
            else:
                if not document[feat]:
                    s += '0' + '\t'
                else:
                    s += str(document[feat]) + '\t'
        to_print += s[:-1] + '\n'

    return to_print
