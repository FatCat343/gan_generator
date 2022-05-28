import torchdata.datapipes as dp
import json


class SnliLoader:
    def snli_train_all(self):
        file = dp.iter.FileOpener(['.root/snli_1.0/snli_1.0_train.jsonl']) \
            .readlines(decode=True, return_path=False, strip_newline=True) \
            .map(lambda line: json.loads(line.strip())) \
            .map(lambda line: (line['sentence1'], line['sentence2']))
        return file

    def snli_valid_option(self, option):
        file = dp.iter.FileOpener(['.root/snli_1.0/snli_1.0_dev.jsonl']) \
            .readlines(decode=True, return_path=False, strip_newline=True) \
            .map(lambda line: json.loads(line.strip())) \
            .filter(lambda line: line['gold_label'] == option) \
            .map(lambda line: (line['sentence1'], line['sentence2']))
        return file

    def snli_train_option(self, option):
        file = dp.iter.FileOpener(['.root/snli_1.0/snli_1.0_train.jsonl']) \
            .readlines(decode=True, return_path=False, strip_newline=True) \
            .map(lambda line: json.loads(line.strip())) \
            .filter(lambda line: line['gold_label'] == option) \
            .map(lambda line: (line['sentence1'], line['sentence2']))
        return file
