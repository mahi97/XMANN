"""Base Task"""

from attr import attrs


class BaseTask(object):
    def __init__(self):
        self.model = TaskModel
        self.param = TaskParam


@attrs
class TaskParam(object):
    pass


@attrs
class TaskModel(object):
    pass
