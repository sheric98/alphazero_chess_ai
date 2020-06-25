import numpy as np


pieces_index = {'K': 0, 'Q': 1, 'R': 2, 'B': 3, 'N': 4, 'P': 5,
                'k': 6, 'q': 7, 'r': 8, 'b': 9, 'n': 10, 'p': 11}


castle_str = 'KQkq'


def get_coord(alg):
    row = int(alg[1]) - 1
    col = ord(alg[0]) - ord('a')
    return row, col


def get_alg(coord):
    ind = str(coord[0] + 1)
    char = chr(ord('a') + coord[1])
    return char + ind


def flip_move(uci_move):
    move = list(uci_move)
    move[1] = str(9 - int(move[1]))
    move[3] = str(9 - int(move[3]))
    return ''.join(move)


def encode_ep(ep):
    encoded = np.zeros((1, 8, 8), dtype=np.float32)
    if ep != '-':
        row, col = get_coord(ep)
        encoded[0][row][col] = 1
    return encoded


def encode_castles(castle):
    boards = []
    for char in castle_str:
        fill = int(char in castle)
        enc = np.full((8, 8), fill, dtype=np.float32)
        boards.append(enc)
    return np.stack(boards)


def encode_num(num):
    encoded = np.full((1, 8, 8), num, dtype=np.float32)
    return encoded


def encode_pieces(board):
    encoded = np.zeros((12, 8, 8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            coord = 8*i + j
            piece = board.piece_at(coord)
            if piece is not None:
                ind = pieces_index[piece.symbol()]
                encoded[ind][i][j] = 1
    return encoded


def encode_board(board):
    fen = board.fen()
    parts = fen.split(' ')
    castles = parts[2]
    ep = parts[3]
    half_moves = int(parts[4])

    enc_pieces = encode_pieces(board)
    enc_castles = encode_castles(castles)
    enc_ep = encode_ep(ep)
    enc_halves = encode_num(half_moves)

    enc_boards = (enc_pieces, enc_castles, enc_ep, enc_halves)

    return np.vstack(enc_boards)


### Move Encoding ###

def get_diff(alg1, alg2):
    row1, col1 = get_coord(alg1)
    row2, col2 = get_coord(alg2)
    return row2 - row1, col2 - col1


# check for pawn promotion
def is_promotion(uci_move):
    return len(uci_move) == 5

# map diffs to plane and back
def generate_diff_to_plane_and_back():
    # Queen Moves
    diff_to_plane = {}
    plane_to_diff = {}
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
            (1, -1), (1, 0), (1, 1), (0, 1)]
    for i, direc in enumerate(dirs):
        for j in range(7):
            dir_tup = (direc[0] * (j+1), direc[1] * (j+1))
            plane = 7*i + j
            diff_to_plane[dir_tup] = plane
            plane_to_diff[plane] = dir_tup

    # Knight Moves
    knight_diffs = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                    (1, -2), (1, 2), (2, -1), (2, 1)]
    for i, diff in enumerate(knight_diffs):
        plane = 56 + i
        diff_to_plane[diff] = plane
        plane_to_diff[plane] = diff

    # pawn under promotions
    pawn_str = 'qnbr'
    for i, char in enumerate(pawn_str):
        for j in range(3):
            horiz_diff = j - 1
            tup = (horiz_diff, char)
            plane = 64 + (3*i) + j
            diff_to_plane[tup] = plane
            plane_to_diff[plane] = tup

    return diff_to_plane, plane_to_diff


diff_to_plane, plane_to_diff = generate_diff_to_plane_and_back()


def get_plane_ind(uci_move):
    diff1, diff2 = get_diff(uci_move[:2], uci_move[2:4])
    if is_promotion(uci_move):
        char = uci_move[4]
        tup = (diff2, char)
    else:
        tup = (diff1, diff2)
    return diff_to_plane[tup]


def get_prob_mask(uci_moves):
    mask = np.zeros((8, 8, 76), dtype=int)
    for uci_move in uci_moves:
        start = uci_move[:2]
        row, col = get_coord(start)
        plane = get_plane_ind(uci_move)
        mask[row][col][plane] = 1
    return mask.flatten()


def decode_dir(plane):
    return plane_to_diff[plane]
