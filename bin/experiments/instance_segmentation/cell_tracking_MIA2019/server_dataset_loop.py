
import Pyro4
Pyro4.config.COMPRESSION = True
Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED = {'pickle'}
import socket
from datasets.pyro_dataset import PyroServerDataset

from bin.experiments.instance_segmentation.cell_tracking_MIA2019.dataset import Dataset


@Pyro4.expose
class CellTrackingServerDataset(PyroServerDataset):
    def __init__(self):
        super(CellTrackingServerDataset, self).__init__(queue_size=8, refill_queue_factor=0.0, n_threads=8, use_multiprocessing=False)

    def init_with_parameters(self, *args, **kwargs):
        # TODO: adapt base folder, in case this script runs on a remote server
        #kwargs['base_folder'] = os.path.join('ADAPT_TO_CORRECT_BASE_FOLDER', kwargs['dataset_name'])
        self.dataset_class = Dataset(*args, **kwargs)
        self.dataset = self.dataset_class.dataset_train()


if __name__ == '__main__':
    print('start')
    daemon = Pyro4.Daemon(host=socket.gethostname(), port=47832)
    print(daemon.register(CellTrackingServerDataset(), 'cell_tracking'))
    daemon.requestLoop()
