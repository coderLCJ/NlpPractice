



def open_file(path, sep=' '):
    """读取文件"""
    src = []
    tgt = []
    with open(path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f.readlines()):    # 
            line = line.strip().split(sep)
            tmp_src = str(line[0])
            tmp_tgt = str(line[1])
            # 若文本和标签都非空
            if tmp_src and tmp_tgt:
                src.append(tmp_src)
                tgt.append(tmp_tgt)
    return src, tgt



def write_file(word2index, path):
    """写文件"""
    with open(path, 'w', encoding='utf8') as f:
        for k,v in word2index.items():
            string = k + ' ' + str(v) + '\n'
            f.write(string)


def write_text(text, path):
    """写文件"""
    with open(path, 'w', encoding='utf8') as f:
        for x in text:
            string = str(x) + '\n'
            f.write(string)
            
            
            