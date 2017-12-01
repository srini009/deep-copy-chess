import sys
import time

import chess.pgn

from chess import SQUARES_180, BB_SQUARES, BB_FILE_H, H1


def board_to_string(board):
    builder = []
    for square in SQUARES_180:
        piece = board.piece_at(square)
        if piece:
            builder.append(piece.symbol())
        else:
            builder.append(".")
    return "".join(builder)


class GameConvertor(chess.pgn.BaseVisitor):
    def __init__(self):
        self.boards = []
        self.valid = True

    def visit_move(self, board, move):
        if self.valid:
            self.boards.append(board_to_string(board))

    def to_string(self):
        return "\n".join(self.boards)


def load_games(pgn_file_name):
    with open(pgn_file_name) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game:
                yield game


def main():
    if len(sys.argv) < 3:
        print("Usage: {} <path_to_pgn> <path_to_output_file>".format(sys.argv[0]))
        sys.exit(1)

    output_dir_path = sys.argv[2]
    batch_count = 0
    total_count = 0
    start_time = time.time()
    total_start_time = time.time()
    file_count = 1
    output_file_name = "{}/{}.txt".format(output_dir_path, file_count)
    output_file = open(output_file_name, "w")
    for game in load_games(sys.argv[1]):
        try:
            convertor = GameConvertor()
            game.accept(convertor)
            if not convertor.valid:
                continue
            batch_count += 1
            output_file.write(convertor.to_string())
            output_file.write("\n")
            if batch_count == 10000:
                output_file.close()
                file_count += 1
                output_file_name = "{}/{}.txt".format(output_dir_path, file_count)
                output_file = open(output_file_name, "w")
                elapsed_time = (time.time() - start_time)
                print("Converted {} games in {:.2f} seconds".format(batch_count, elapsed_time))
                total_count += batch_count
                batch_count = 0
                start_time = time.time()
        except Exception as e:
            print("Exception occured  \"{}\". Continuing")
    output_file.close()
    if batch_count != 10000:
        total_count += batch_count
    elapsed_time = (time.time() - total_start_time)
    print("Totally converted games {} in {:.2f} seconds".format(total_count, elapsed_time))

if __name__ == '__main__':
    main()
