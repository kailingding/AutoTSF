import abc


class ModelInterface:
    '''
            A model interface
    '''

    @abc.abstractmethod
    def __init__(self, is_model_trained=False):
        self.is_model_trained = is_model_trained  # a training check flag

    @abc.abstractmethod
    def tune_hyperparameters(self, *args, **kwargs):
        '''Derived class must implement this method'''
        raise NotImplementedError()

    @abc.abstractmethod
    def train_model(self, X, y, **kwargs):
        '''Derived class must implement this method'''
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate_model(self, *args, **kwargs):
        """"Derived class must implement this method"""
        raise NotImplementedError()

    @abc.abstractmethod
    def precit(self, num_steps, *args, **kwargs):
        """"Derived class must implement this method"""
        """Return the length of num_steps forecast"""
        raise NotImplementedError()

    @property
    def is_model_trained(self):
        return self.__is_model_trained

    @is_model_trained.setter
    def is_model_trained(self, is_model_trained):
        assert is_model_trained in [True, False, None], \
            'is_model_trained must be either True, False, or None'
        self.__is_model_trained = is_model_trained
