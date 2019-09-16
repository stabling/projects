"""Maintain info for saved models. """

import os


class Info:

    def __init__(self, param_dir):
        self.param_dir = param_dir
        self.info_pth = os.path.join(param_dir, "model.info")
        self._info = []
        if os.path.exists(self.info_pth):
            with open(self.info_pth) as info_file:
                for _info in info_file.readlines():
                    _param_pth, _metric = _info.split()
                    self._info.append((_param_pth, float(_metric)))
        else:
            os.mknod(self.info_pth)

    def __len__(self):
        return len(self._info)

    def __getitem__(self, item):
        if -len(self._info) <= item < len(self._info):
            _param_pth, _metric = self._info[item]
        else:
            _param_pth, _metric = None, float('inf')

        return _param_pth, _metric

    def write(self, param_pth, metric):
        with open(self.info_pth, mode='a') as info_file:
            info_file.write(f"{param_pth} {metric}\n")
        self._info.append((param_pth, metric))
