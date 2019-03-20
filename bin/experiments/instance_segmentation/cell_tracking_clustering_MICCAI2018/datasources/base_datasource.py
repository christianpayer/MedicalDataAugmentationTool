
class BaseDatasource(object):
    def __init__(self, id_dict_preprocessing):
        self.id_dict_preprocessing = id_dict_preprocessing

    def get(self, *args, **kwargs):
        raise NotImplementedError

    def preprocess_id_dict(self, id_dict):
        if self.id_dict_preprocessing is not None:
            return self.id_dict_preprocessing(id_dict)
        return id_dict