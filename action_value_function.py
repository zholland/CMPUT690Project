import abc


class AbstractActionValueFunction(metaclass=abc.ABCMeta):

    def action_values(self, S):
        """"""

    def value(self, S, A):
        """"""

    def update(self, S, A, new_value):
        """"""
