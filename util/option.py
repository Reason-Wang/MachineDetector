import argparse


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()


    def print_options(self):
        raise NotImplementedError

    def initialize_parser(self):
        self.parser.add_argument('--name', type=str, help='name of the experiment')
        self.parser.add_argument('--model_name_or_path', type=str, help='name or path of the model')
        # self.parser.add_argument('--dataset', type=str, help='name of the dataset')
        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--seed', type=int, default=42)
        # self.parser.add_argument('--input_size', type=int, default=1024)
        self.parser.add_argument('--no_shuffle_train', action='store_true')
        self.parser.add_argument('--target_size', type=int, default=1)
        self.parser.add_argument('--scheduler', type=str, default='cosine')
        self.parser.add_argument('--num_cycles', type=float, default=0.5)
        self.parser.add_argument('--num_workers', type=int, default=0)
        self.parser.add_argument('--epochs', type=int, default=3)
        self.parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
        self.parser.add_argument('--warmup_ratio', type=int, default=0.01)
        self.parser.add_argument('--gradient_clipping', action='store_true')
        self.parser.add_argument('--max_grad_norm', type=float, default=1.0)
        self.parser.add_argument('--lr', type=float, default=2e-5)
        self.parser.add_argument('--max_len', type=int, default=512)
        self.parser.add_argument('--fc_dropout', type=float, default=0.1)
        self.parser.add_argument('--num_samples', type=int, default=None)
        self.parser.add_argument('--apex', action='store_true')
        self.parser.add_argument('--target_size', type=int, default=2, help='Target size.')


    def parse(self):
        opt = self.parser.parse_known_args()
        return opt

    def get_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        return message