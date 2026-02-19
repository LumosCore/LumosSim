def calculate_test_result_statistics(result_file, max_num):
    with open(result_file, "r") as f:
        res = []
        i = 0
        for line in f:
            if line == '1.0\n' or line == 'nan\n':
                continue
            res.append(float(line) - 0.01)
            i += 1
            if i == max_num:
                break
    return res


if __name__ == '__main__':
    figret = calculate_test_result_statistics('Figret/result_2200_500_0.1.txt', 1048)
    print(sum(figret) / len(figret), max(figret))
    jupiter = calculate_test_result_statistics('Jupiter/result.txt', 1048)
    print(sum(jupiter) / len(jupiter), max(jupiter))
