class DictToObj:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def dict_list_append(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)
