import lzma, json, copy


# lựa chọn 8k từ tiếng Việt để bổ xung vào vocab
vi_words = []
for idx, line in enumerate(lzma.open("data/vi_words_impact_phi.jsonl.xz")):
    word = json.loads(line)["word"]
    if word[0] != " ": continue
    word = '▁' + word[1:]
    word = word.replace('_', '▁')
    vi_words.append( word )
    if len(vi_words) == 8000: break

print(vi_words[-3:], len(vi_words))

if __name__ == "__main__":

    # https://huggingface.co/nbroad/donut-base-ascii/blob/main/remove-donut-tokens.ipynb
    from transformers.convert_slow_tokenizer import import_protobuf

    model_pb2 = import_protobuf()
    m = model_pb2.ModelProto()

    filename = "data/phi_tokenizer.model.xz"
    m.ParseFromString(lzma.open(filename, 'rb').read())

    print(len(m.pieces)) # 32k
    print(m.pieces[:3])

    vocab = set([p.piece for p in m.pieces])

    for word in vi_words:
        if word not in vocab:
            piece = copy.copy(m.pieces[-1])
            piece.piece = word
            piece.score = -50000
            m.pieces.append(piece)

    new_filename = "phi_tokenizer__extend_vocab.model"
    with open(new_filename, 'wb') as f:
        f.write(m.SerializeToString())


    #########
    import sentencepiece as spm
    from pyvi import ViTokenizer

    sp = spm.SentencePieceProcessor(model_file=new_filename)

    text = """
    Ông Vũ Minh Đức, Cục trưởng Cục Nhà giáo và Cán bộ quản lý giáo dục, ngày 11/10, trả lời VnExpress về một số đề xuất mới của dự thảo Luật Nhà giáo - sẽ được trình Quốc hội thảo luận tại kỳ họp tới (khai mạc 21/10).

    - Theo dự thảo Luật Nhà giáo, Bộ Giáo dục và Đào tạo đề xuất miễn học phí cho con giáo viên, từ mầm non đến đại học. Vì sao Bộ đưa ra đề xuất này?

    - Dự thảo Luật Nhà giáo được xây dựng với 5 chính sách quan trọng gồm: định danh nhà giáo; tiêu chuẩn và chức danh nhà giáo; tuyển dụng, sử dụng và chế độ làm việc của nhà giáo; đào tạo, bồi dưỡng, đãi ngộ và tôn vinh nhà giáo; quản lý nhà nước về nhà giáo. Các chính sách này đã được Quốc hội, Chính phủ thông qua.

    Từ khi công bố dự thảo hồi tháng 5 đến nay, chúng tôi có những điều chỉnh nhưng vẫn bám sát 5 chính sách đó. Miễn học phí cho con em giáo viên là một trong những đề xuất liên quan đến đãi ngộ với nhà giáo, xuất phát từ một số lý do.
    """

    pyvi = ViTokenizer.tknz(text, allowed_words = vi_words)

    tids = sp.encode(pyvi)
    tokens = [ sp.decode(x) for x in tids ]

    print(text)
    print(pyvi)
    print(tokens)

    vocab = [[sp.id_to_piece(id), id] for id in range(sp.get_piece_size())]
    pieces = [ x[0] for x in vocab ]
    assert "▁Cục▁trưởng" in pieces

    print(sp.vocab_size())
    print(vocab[-10:])
