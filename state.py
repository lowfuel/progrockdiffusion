import json5 as json
from typing import Union


class State(dict):
    def __init__(self, d={}):
        if isinstance(d, str):
            with open(d) as fp:
                try:
                    d = json.load(fp)
                except Exception as e:
                    print('Failed to open or parse ' + d + ' - Check formatting.')
                    raise e
        super().__init__({**d})

    def __getattr__(self, k):
        try:
            return self[k]
        except:
            return None

    def __getitem__(self, k):
        try:
            return super().__getitem__(k)
        except:
            return None

    def __setattr__(self, k, v):
        self.update({k: v})

    def __add__(self, other: Union[dict, 'State']):
        return State({**self, **other})

    def __or__(self, other):
        return State({**self, **other})

    def __sub__(self, other: Union[dict, 'State', str, list, tuple]):
        s = State(self)
        if isinstance(other, str):
            other = [other]
        for k in other:
            try:
                del s[k]
            except:
                pass
        return s
