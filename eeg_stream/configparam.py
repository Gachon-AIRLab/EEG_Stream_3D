
class ConfigParam:

    data_path = ''
    weight_path = ''
    weight_prefix = ''
    result_path = ''
    use_pretrained = False
    use_predefined_idx = False
    pretrained_name = ''
    train_ratio = 0.0
    split_path = ''
    target_label = 'valence'

    num_subject = 0
    num_video = 0
    num_channel = 0
    dataset_type = ''
    target_subject = []
    target_video = []
    train_ratio = 0
    exception_subject = []
    exception_video = []

    train_method = ''
    learning_ratio = 0
    num_epoch = 0
    batch_size = 0

    def __init__(self):
        return

    def LoadConfiguration(self, run_path):

        f = open(run_path, 'r')

        # print('- Load configuration -')
        while True:
            line = f.readline()
            if not line: break
            words = line.replace('\n','').split(': ')
            if len(words) > 1:
                if words[0] == 'data_path':
                    self.data_path = words[1]
                elif words[0] == 'result_path':
                    self.result_path = words[1]
                    self.weight_path = self.result_path + 'pretrained/'
                    self.split_path = self.result_path + 'split/'
                elif words[0] == 'weight_prefix':
                    self.weight_prefix = words[1]
                elif words[0] == 'use_pretrained':
                    self.use_pretrained = bool(int(words[1]))
                elif words[0] == 'use_predefined_idx':
                    self.use_predefined_idx = bool(int(words[1]))
                elif words[0] == 'pretrained_name':
                    self.pretrained_name = words[1]
                elif words[0] == 'num_subject':
                    self.num_subject = int(words[1])
                elif words[0] == 'num_video':
                    self.num_video = int(words[1])
                elif words[0] == 'num_channel':
                    self.num_channel = int(words[1])
                elif words[0] == 'dataset_type':
                    self.dataset_type = words[1]
                elif words[0] == 'target_subject':
                    self.target_subject = words[1].split(' ')
                    self.target_subject = list(map(int, self.target_subject))
                elif words[0] == 'target_video':
                    self.target_video = words[1].split(' ')
                    self.target_video = list(map(int, self.target_video))
                elif words[0] == 'train_ratio':
                    self.train_ratio = float(words[1])
                elif words[0] == 'exception_subject':
                    self.exception_subject = words[1].split(' ')
                    self.exception_subject = list(map(int, self.exception_subject))
                elif words[0] == 'exception_video':
                    self.exception_video = words[1].split(' ')
                    self.exception_video = list(map(int, self.exception_video))
                elif words[0] == 'train_method':
                    self.train_method = words[1]
                elif words[0] == 'target_label':
                    self.target_label = words[1]
                elif words[0] == 'learning_rate':
                    self.learning_rate = float(words[1])
                elif words[0] == 'num_epoch':
                    self.num_epoch = int(words[1])
                elif words[0] == 'batch_size':
                    self.batch_size = int(words[1])


        f.close()

    def PrintConfig(self):
        print('data_path = %s'% self.data_path)
        print('result_path = %s'% self.result_path)
        print('weight_path = %s' % self.weight_path)
        print('weight_prefix = %s'% self.weight_prefix)
        print('use_pretrained = %d' % self.use_pretrained)
        print('use_predefined_idx = %d' % self.use_predefined_idx)
        if self.use_pretrained:
            print('weight_prefix = %s' % self.pretrained_name)
        print('split_path = %s' % self.split_path)
        print('num_subject = %d'% self.num_subject)
        print('num_video = %d'% self.num_video)
        print('num_channel = %d'% self.num_channel)
        print('dataset type = %s' % self.dataset_type)
        if 'kfold' in self.dataset_type:
            if (len(self.target_subject) > 0):
                print('target_subject = ', end=" ")
                if (self.target_subject[0] == 0):
                    print('all')
                else:
                    print(self.target_subject)
            if (len(self.target_video) > 0):
                print('target_video = ', end=" ")
                if (self.target_video[0] == 0):
                    print('all')
                else:
                    print(self.target_video)
            print('train_ratio = %f' % self.train_ratio)
        if 'loo' in self.dataset_type:
            if (len(self.exception_subject) > 0):
                print('exception_subject = ', end=" ")
                if (self.exception_subject[0] == 0):
                    print('all')
                else:
                    print(self.exception_subject)
            if (len(self.exception_video) > 0):
                print('exception_video = ', end=" ")
                if (self.exception_video[0] == 0):
                    print('all')
                else:
                    print(self.exception_video)

        print('train_method = %s' % self.train_method)
        print('target_label = %s' % self.target_label)
        print('learning_rate = %f' % self.learning_rate)
        print('num_epoch = %d'% self.num_epoch)
        print('batch_size = %d'% self.batch_size)
        return