import os

def split_pgn_file(input_path, output_dir, games_per_file=100):
    """
    Chia nhỏ file PGN lớn thành nhiều file nhỏ hơn, mỗi file chứa games_per_file ván cờ.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    game_buffer = []
    file_count = 1
    game_count = 0
    for line in lines:
        game_buffer.append(line)
        if line.strip() == '' and any(l.startswith('[Event') for l in game_buffer):
            game_count += 1
            if game_count % games_per_file == 0:
                out_path = os.path.join(output_dir, f'split_{file_count}.pgn')
                with open(out_path, 'w', encoding='utf-8') as out_f:
                    out_f.writelines(game_buffer)
                print(f'Đã tạo {out_path} chứa {games_per_file} ván cờ.')
                file_count += 1
                game_buffer = []
    # Ghi phần còn lại
    if game_buffer:
        out_path = os.path.join(output_dir, f'split_{file_count}.pgn')
        with open(out_path, 'w', encoding='utf-8') as out_f:
            out_f.writelines(game_buffer)
        print(f'Đã tạo {out_path} chứa {game_count % games_per_file} ván cờ.')

if __name__ == "__main__":
    split_pgn_file("h:/tu/pgn_example.pgn", "h:/tu/pgn_split", games_per_file=100)
