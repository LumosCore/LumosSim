def get_empty_num(opt_file):
    with open(opt_file, 'r') as f:
        lines = f.readlines()
    empty_num = 0
    for line in lines:
        if line.strip() == '0.0':
            empty_num += 1
    return empty_num


if __name__ == '__main__':
    opt_files = ['train/48.opt', 'train/49.opt', 'train/50.opt', 'test/48.opt']
    for opt_file in opt_files:
        print(get_empty_num(opt_file))
