import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim

import rnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_token = 'G'
end_token = 'E'
batch_size = 64


def process_poems1(file_name):
    """

    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                # 句子中含有非法字符的筛选去
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                # 句子不符合限制的筛选去
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                print(" ")
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频，格式为{'word':count}
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序，词频由高到低
    # 将其拆分成 两个元组：一个是字的集合，一个是词频值的集合
    words, _ = zip(*count_pairs)
    # 在words数组后面加上一个空格
    words = words[:len(words)] + (' ',)
    # 为每一个汉字添加一个索引，索引顺序是按顺序来的（0~len）
    word_int_map = dict(zip(words, range(len(words))))
    '''
    map(word_int_map.get, poem) 作用：将 poem 中每个字转换成索引。
    '''
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems] # 将诗句中的每一个字转换成一个索引
    return poems_vector, word_int_map, words




def process_poems2(file_name):
    """
    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        # content = ''
        for line in f.readlines():
            try:
                line = line.strip()
                if line:
                    content = line.replace(' '' ', '').replace('，','').replace('。','')
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                    start_token in content or end_token in content:
                        continue
                    if len(content) < 5 or len(content) > 80:
                        continue
                    # print(content)
                    content = start_token + content + end_token
                    poems.append(content)
                    # content = ''
            except ValueError as e:
                # print("error")
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size # 34813 // 100 = 348
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y  = row[1:]
            y.append(row[-1])
            y_data.append(y)
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training():
    # 处理数据集
    poems_vector, word_to_int, vocabularies = process_poems1('./poems.txt')
    # 生成batch
    print("finish  loadding data")
    BATCH_SIZE = 100

    torch.manual_seed(5)
    word_embedding = rnn.word_embedding( vocab_length= len(word_to_int) + 1 , embedding_dim= 100)
    rnn_model = rnn.RNN_model(batch_sz = BATCH_SIZE,vocab_len = len(word_to_int) + 1 ,word_embedding = word_embedding ,embedding_dim= 100, lstm_hidden_dim=128)

    optimizer=optim.RMSprop(rnn_model.parameters(), lr=0.01)

    loss_fun = torch.nn.NLLLoss() # CrossEntropyLoss() = +softmax()+log() + NLLLoss()
    # NLLLoss 主要用于 分类任务，常和 softmax 结合使用：
    # 如果网络的最后一层是 log_softmax（对数 Softmax），则用 NLLLoss()
    # 如果最后一层直接输出的结果，则应该使用 CrossEntropyLoss()，因为这个函数自带 softmax 操作。
    
    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))  # if you have already trained your model you can load it by this line.

    for epoch in range(30):
        # 已经把要输入的数据分成了348个batch，每个batch有100个数据
        # 需要注意的是，batch_output指的是每个x输入数据的下一个数据（即“日照香炉生紫烟中”，输入“日”，预测“照”）
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        for batch in range(n_chunk):
            # 每个batch里面有100个数据
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch] # (batch , time_step)
            loss = 0
            # 遍历每个batch中的100个数据
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype = np.int64)
                y = np.array(batch_y[index], dtype = np.int64)
                x = Variable(torch.from_numpy(np.expand_dims(x,axis=1)))
                y = Variable(torch.from_numpy(y ))
                pre = rnn_model(x)
                loss += loss_fun(pre , y)
                if index == 0:# 只在 index == 0（即 batch 内第一个样本）时打印预测结果，方便监测训练情况
                    _, pre = torch.max(pre, dim=1)
                    print('prediction', pre.data.tolist()) 
                    print('b_y       ', y.data.tolist())   
                    print('*' * 30)
            loss  = loss  / BATCH_SIZE
            print("epoch  ",epoch,'batch number',batch,"loss is: ", loss.data.tolist())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(rnn_model.parameters(), 1) # 限制梯度的 L2 范数，使其不超过 1
            optimizer.step()

            if batch % 20 ==0:
                torch.save(rnn_model.state_dict(), './poem_generator_rnn')
                print("finish  save model")



def to_word(predict, vocabs):  # 预测的结果转化成汉字
    sample = np.argmax(predict)

    if sample >= len(vocabs):
        sample = len(vocabs) - 1

    return vocabs[sample]


def pretty_print_poem(poem):  # 令打印的结果更工整
    shige=[]
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 1:
            print(s + '。')


def gen_poem(begin_word):
    poems_vector, word_int_map, vocabularies = process_poems1('./poems.txt')
    
    word_embedding = rnn.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100).to(device)
    rnn_model = rnn.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,
                               embedding_dim=100, lstm_hidden_dim=128).to(device)

    # 修改加载模型的位置
    rnn_model.load_state_dict(torch.load('./poem_generator_rnn', map_location=torch.device('cpu')))
    rnn_model.to(device)
    rnn_model.eval()

    poem = begin_word
    word = begin_word
    while word != end_token:
        input_seq = np.array([word_int_map[w] for w in poem], dtype=np.int64)
        input_tensor = Variable(torch.from_numpy(input_seq)).to(device)
        
        # 传入当前诗歌的数值表示，预测下一个字。
        output = rnn_model(input_tensor, is_test=True)

        # 取最后一个时间步的预测结果（因为 RNN 处理的是序列）。
        word = to_word(output.data.tolist()[-1], vocabularies)
        poem += word

        # 防止 无限生成，限制最大长度 30。
        if len(poem) > 30:
            break
    return poem




# run_training()  # 如果不是训练阶段 ，请注销这一行 。 网络训练时间很长。



pretty_print_poem(gen_poem("日"))
pretty_print_poem(gen_poem("红"))
pretty_print_poem(gen_poem("山"))
pretty_print_poem(gen_poem("夜"))
pretty_print_poem(gen_poem("色"))
pretty_print_poem(gen_poem("山"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("君"))

