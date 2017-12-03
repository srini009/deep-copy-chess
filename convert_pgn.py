import sys
import multiprocessing
import time

import chess.pgn

from chess import SQUARES_180, BB_SQUARES, BB_FILE_H, H1


BATCH_SIZE = 1000


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
        return "".join(self.boards)


def load_games(pgn_file_name):
    with open(pgn_file_name) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game:
                yield game
            else:
                return


def convert_pgn(args):
    prefix, pgn_file_name, output_dir_path = args
    batch_count = 0
    total_count = 0
    file_count = 1
    output_file_name = None
    for game in load_games(pgn_file_name):
        try:
            if not output_file_name:
                output_file_name = "{}/{}-{}.txt".format(output_dir_path, prefix, file_count)
                output_file = open(output_file_name, "w")
            convertor = GameConvertor()
            game.accept(convertor)
            if not convertor.valid:
                continue
            batch_count += 1
            output_file.write(convertor.to_string())
            output_file.write("\n")
            if batch_count == BATCH_SIZE:
                output_file.close()
                output_file_name = None
                file_count += 1
                total_count += batch_count
                batch_count = 0
        except Exception as e:
            print("Exception occured  \"{}\". Continuing".format(e))
    output_file.close()
    if batch_count != 0:
        total_count += batch_count
    return total_count


def main():
    if len(sys.argv) < 4:
        print("Usage: {} <path_to_pgn1> ...... <path_to_output_dir>".format(sys.argv[0]))
        sys.exit(1)

    start_time = time.time()
    convert_args = [(i, sys.argv[i], sys.argv[-1]) for i in range(1, len(sys.argv) - 1)]
    pool = multiprocessing.Pool(8)
    counts = pool.map(convert_pgn, convert_args)
    elapsed_time = time.time() - start_time
    total_count = sum(counts)
    rate = total_count / elapsed_time
    print("Converted {} games in {:.2f} seconds. Average rate = {:.3f} games/second".format(total_count, elapsed_time, rate))

if __name__ == '__main__':
    main()
