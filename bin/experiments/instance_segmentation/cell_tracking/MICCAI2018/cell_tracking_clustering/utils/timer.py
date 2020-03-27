import datetime


class Timer(object):
    def __init__(self,
                 name,
                 print_on_start=True,
                 print_on_stop=True):
        self.name = name
        self.print_on_start = print_on_start
        self.print_on_stop = print_on_stop
        self.start_time = None
        self.stop_time = None

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.start_time = datetime.datetime.now()
        if self.print_on_start:
            self.print_start()

    def stop(self):
        self.stop_time = datetime.datetime.now()
        if self.print_on_stop:
            self.print_stop()

    def elapsed_time(self):
        return self.stop_time - self.start_time

    def time_string(self, time):
        return '%02d:%02d:%02d' % (time.hour, time.minute, time.second)

    def seconds_string(self, duration):
        return '%d.%03d' % (duration.seconds, duration.microseconds // 1000)

    def print_start(self):
        print(self.name, 'starting at', self.time_string(self.start_time))

    def print_stop(self):
        print(self.name, 'finished at', self.time_string(self.stop_time), 'elapsed time', self.seconds_string(self.elapsed_time()))
