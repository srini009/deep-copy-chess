import chess
import chess.svg
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


def string_to_board(combined_board):
    board = chess.Board()
    board.reset_board()
    board_lines = textwrap.wrap(combined_board, 8)
    for r, row_line in enumerate(board_lines):
        for c, piece_str in enumerate(row_line):
            index = (7 - r) * 8 + c
            piece = None if piece_str == "." else chess.Piece.from_symbol(piece_str)
            board.set_piece_at(index, piece)
    return board


def board_to_svg(board, file_name):
    svg = chess.svg.board(board=board, size=800)
    with open(file_name, 'w') as svg_file:
        svg_file.write(svg)


def main():
    if len(sys.argv) < 3:
        print("Usage: {} <board_str> <svg_file_name>".format(sys.argv[0]))
        sys.exit(1)
    board_to_svg(string_to_board(sys.argv[1]), sys.argv[2])


if __name__ == '__main__':
    main()
