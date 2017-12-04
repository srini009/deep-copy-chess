import collections
import os
import sys
import textwrap


def count_unique_boards(dir_path):
    file_names = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
    boards = collections.Counter()
    num_lines = 0
    for file_name in file_names:
        with open(file_name) as games_file:
            content = games_file.read()
            lines = content.split('\n')
            for line in lines:
                if not line:
                    continue
                num_lines  += 1
                positions = textwrap.wrap(line, 64)
                print(len(positions))
                for board in positions:
                    boards[board] += 1
    return sum(boards.values())

def main():
    if len(sys.argv) < 2:
        print("Usage: {} <path_of_txts>".format(sys.argv[1]))
        sys.exit(1)
    dir_path = sys.argv[1]
    print(count_unique_boards(dir_path))


if __name__ == '__main__':
    main()
