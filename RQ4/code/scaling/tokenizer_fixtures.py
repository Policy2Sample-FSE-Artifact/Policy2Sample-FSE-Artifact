"""Small deterministic tokenizer fixture used by the adapter microbenchmark."""


class GreedyPieceTokenizer:
    def __init__(self, pieces: tuple[str, ...]) -> None:
        self.pieces = tuple(sorted(set(pieces), key=lambda piece: (-len(piece), piece)))
        self.ids = {piece: index + 1 for index, piece in enumerate(self.pieces)}
        self.id_to_piece = {value: key for key, value in self.ids.items()}

    def encode(self, text: str) -> tuple[int, ...]:
        output = []
        offset = 0
        while offset < len(text):
            piece = next((part for part in self.pieces if text.startswith(part, offset)), None)
            if piece is None:
                piece = text[offset]
                if piece not in self.ids:
                    self.ids[piece] = len(self.ids) + 1
                    self.id_to_piece[self.ids[piece]] = piece
                output.append(self.ids[piece])
                offset += 1
            else:
                output.append(self.ids[piece])
                offset += len(piece)
        return tuple(output)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self.id_to_piece[token_id] for token_id in token_ids)
