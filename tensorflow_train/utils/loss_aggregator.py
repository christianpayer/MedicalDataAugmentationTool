
import csv
import datetime
import os

import tensorflow.compat.v1 as tf

from tensorflow_train.utils.tensorflow_util import create_reset_metric


class LossAggregator(object):
    def __init__(self, session, loss_dict, summary, summary_placeholders, name, summary_folder, csv_filename):
        self.session = session
        self.loss_dict = loss_dict
        self.name = name
        self.summary_folder = summary_folder
        self.csv_filename = csv_filename
        self.loss_metrics = {}
        self.summary = summary
        self.summary_placeholders = summary_placeholders
        for key, value in sorted(loss_dict.items()):
            self.loss_metrics[key] = create_reset_metric(tf.metrics.mean, key + '_' + name, values=value, name=name + '/' + key)
        self.summary_writer = tf.summary.FileWriter(summary_folder, self.session.graph)
        self.last_finalize_time = datetime.datetime.now()

    def get_update_ops(self):
        update_op_list = []
        for key in sorted(self.loss_dict.keys()):
            update_op_list.append(self.loss_metrics[key][1])
        return tuple(update_op_list)

    def finalize(self, iter, additional_csv_values=None):
        value_op_list = []
        reset_op_list = []
        for key in sorted(self.loss_dict.keys()):
            value_op_list.append(self.loss_metrics[key][0])
            reset_op_list.append(self.loss_metrics[key][2])
        losses = self.session.run(tuple(value_op_list))

        # calculate summary
        summary_feed_dict = {}
        loss_index = 0
        for key in sorted(self.loss_dict.keys()):
            summary_feed_dict[self.summary_placeholders[key]] = losses[loss_index]
            loss_index += 1
        sum = self.session.run(self.summary, feed_dict=summary_feed_dict)
        self.summary_writer.add_summary(sum, iter)

        # calculate time since last finalize
        now = datetime.datetime.now()
        time_since_last_finalize = now - self.last_finalize_time
        self.last_finalize_time = now

        if os.path.exists(self.csv_filename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        with open(self.csv_filename, append_write) as csv_file:
            writer = csv.writer(csv_file)
            row = [str(iter), str(time_since_last_finalize.seconds)]
            for loss in losses:
                row.append(loss)
            if additional_csv_values is not None:
                for value in additional_csv_values:
                    row.append(value)
            writer.writerow(row)

        date_string = '%02d:%02d:%02d' % (now.hour, now.minute, now.second)
        print_string = date_string + ': ' + self.name + ' iter: ' + str(iter) + ' '
        loss_index = 0
        for key in sorted(self.loss_dict.keys()):
            print_string += key + ': ' + str(losses[loss_index]) + ' '
            loss_index += 1
        print_string += 'seconds: %d.%03d' % (time_since_last_finalize.seconds, time_since_last_finalize.microseconds // 1000)
        print(print_string)

        # reset accumulating mean values
        self.session.run(reset_op_list)


class LossSummaryHandler(object):
    def __init__(self, losses):
        self.losses = losses
        self.placeholders = {}
        self.summaries = []
        for loss in self.losses:
            placeholder = tf.placeholder(tf.float32)
            self.placeholders[loss] = placeholder
            summary = tf.summary.scalar(loss, placeholder)
            self.summaries.append(summary)
        self.summary = tf.summary.merge(self.summaries)