import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import re
import os
import io
import time
import random

import seaborn # Attention 시각화
from konlpy.tag import Mecab


# 트렌스포머는 위치에 대한 정보가 없어서
# 사과: (1,5,3,2,7,23,23,....(512개)) + positional
def positional_encoding(pos_len, d_model):
    def cal_angle(position, i):
        return position / np.power(10000, int(i) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, i) for i in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(pos_len)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:,0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:,1::2])
    return sinusoid_table


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)

        self.linear = tf.keras.layers.Dense(d_model)

    def scale_dot_product_attention(self, Q, K, V, mask=None):
        d_k = tf.cast(K.shape[-1], tf.float32)
        # scaled qk 구하기

        QK_T = tf.matmul(Q, K, transpose_b=True)
        scaled_qk = QK_T / tf.math.sqrt(d_k)

        if mask is not None: scaled_qk += (mask * -1e9) # -무한대에서 소프트맥스 -> 0

        attentions = tf.nn.softmax(scaled_qk, axis=-1)
        out = tf.matmul(attentions, V)

        return out, attentions

    def split_heads(self, *xs): # num_heads는 self에 있으니까 받을 필요 없음
        # MultiHead에 넣을려고 분할 
        # x: [ batch x length x embedding_dimension ]
        # return: [ batch x heads x length x embedding_dimension]
        split_xs = []
        for x in xs:
            a,b,c = x.shape

            split_x = tf.reshape(x,(a,b,self.num_heads,self.depth))
            split_x = tf.transpose(split_x, (0,2,1,3))
            split_xs.append(split_x)

        return split_xs


    def combine_heads(self, x):
        # 분할 계산을 마치고, 임베딩을 다시 결합한다.
        # x: [ batch x heads x length x depth ]
        # return: [ batch x length x emb ]
        x = tf.transpose(x,(0,2,1,3))
        a,b,c,d = x.shape

        concat_x = tf.reshape(x, (a,b,c*d))

        return concat_x


    def call(self, Q, K, V, mask=None): #[batch x len x 512 ]
        # 1: Linear_in(Q,K,V) -> WQ, WK, WV
        wq = self.W_q(Q)
        wk = self.W_k(K)
        wv = self.W_v(V)

        # 사과: [1,2,3,4,5,6,6,7,8,9] -> (10,)
        # 사과: [[1,2,3,4,5],[6,6,7,8,9]] -> (2,5) 확실!!
        # apple: 

        # 2: split heads
        W_qkv_split = self.split_heads(wq,wk,wv)

        # 3: scaled dot product attention
        out, attention_weights = self.scale_dot_product_attention(*W_qkv_split, mask)

        # 4: Combine Heads(out) -> out
        out = self.combine_heads(out)

        # 5: Linear_out(out) -> out
        out = self.linear(out)

        return out, attention_weights


class Position_wise_FFN(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super(Position_wise_FFN, self).__init__()
        self.w_1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.w_2 = tf.keras.layers.Dense(d_model)

    def call(self,x):
        out = self.w_1(x)
        out = self.w_2(out)
        return out


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = Position_wise_FFN(d_model,d_ff)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask, training=False):
        # Multi-Head Attention
        residual = x

        out = self.norm1(x)
        out, enc_attn = self.enc_self_attn(out,out,out, mask)
        out = self.dropout(out, training=training)

        out += residual

        # position wise FFN
        residual2 = out
        out = self.norm2(out)
        out = self.ffn(out)
        out = self.dropout(out, training=training)
        out += residual2

        return out, enc_attn



class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads)
        self.dec_attn = MultiHeadAttention(d_model, n_heads)

        self.ffn = Position_wise_FFN(d_model,dff)
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x,y, causality_mask, padding_mask, training=False):
        residual = x
        out = self.norm1(x)
        out, dec_attn = self.dec_self_attn(out,out,out, padding_mask)
        out = self.dropout(out, training=training)
        out += residual

        residual = out
        out = self.norm2(out)
        out, dec_enc_attn = self.dec_attn(out,y,y, causality_mask)
        out = self.dropout(out, training=training)
        out += residual

        residual = out
        out = self.norm3(out)
        out = self.ffn(out)
        out = self.dropout(out, training=training)
        out += residual

        return out, dec_attn, dec_enc_attn


class Encoder(tf.keras.Model):
    def __init__(self,
                 n_layers,
                 d_model,
                 n_heads,
                 dff,
                 dropout):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.enc_layers = [EncoderLayer(d_model,n_heads,dff,dropout) for _ in range(n_layers)]

    def call(self, x, mask, training=False):
        out = x

        enc_attns = []

        for i in range(self.n_layers):
            out, enc_attn = self.enc_layers[i](out, mask, training)
            enc_attns.append(enc_attn)

        return out, enc_attns


class Decoder(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, dff, dropout):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.dec_layers = [DecoderLayer(d_model,n_heads, dff, dropout) for i in range(n_layers)]

    def call(self, x, enc_out, causality_mask, padding_mask, training=False):
        out = x

        dec_attns = []
        dec_enc_attns = []

        for i in range(self.n_layers):
            out, dec_attn, dec_enc_attn = self.dec_layers[i](out,enc_out, causality_mask, padding_mask, training)

            dec_attns.append(dec_attn)
            dec_enc_attns.append(dec_enc_attn)

        return out, dec_attns, dec_enc_attns


def generate_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def generate_causality_mask(src_len, tgt_len):
    mask = 1 - np.cumsum(np.eye(src_len, tgt_len), 0)
    return tf.cast(mask, tf.float32)

def generate_masks(src, tgt):
    enc_mask = generate_padding_mask(src)
    dec_mask = generate_padding_mask(tgt)

    dec_enc_causality_mask = generate_causality_mask(tgt.shape[1], src.shape[1])
    dec_enc_mask = tf.maximum(enc_mask, dec_enc_causality_mask)

    dec_causality_mask = generate_causality_mask(tgt.shape[1], tgt.shape[1])
    dec_mask = tf.maximum(dec_mask, dec_causality_mask)

    return enc_mask, dec_enc_mask, dec_mask


class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(LearningRateScheduler, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)

        return (self.d_model ** -0.5) * tf.math.minimum(arg1, arg2)

learning_rate = LearningRateScheduler(512)
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)



def clean_corpus(kor_path, eng_path):
    with open(kor_path, "r") as f: kor = f.readlines()
    with open(eng_path, "r") as f: eng = f.readlines()
    assert len(kor) == len(eng)
    print(len(kor))

    dataset = set()
    for i,j in zip(kor, eng):
        i = preprocess_sentence(i)
        j = preprocess_sentence(j)
        dataset.add((i,j))
    print(len(dataset))
    cleaned_corpus = list(dataset)
    return cleaned_corpus


def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'([?!,."])', r' \1 ',sentence)
    sentence = re.sub(r'[^A-zㄱ-ㅎㅏ-ㅣ가-힣0-9?!,."]', ' ', sentence)
    sentence = re.sub(r'[" "]+', ' ',sentence)
    sentence = sentence.strip()
    return sentence


import sentencepiece as spm
import os

def generate_tokenizer(corpus,
                       vocab_size,
                       lang="ko",
                       pad_id=0,
                       bos_id=1,
                       eos_id=2,
                       unk_id=3):
    path = 'aiffel/Data/Model/transformer/'
    temp_file = f'{path}corpus_{lang}.temp'

    with open(temp_file, 'w') as f:
        for row in corpus:
            f.write(str(row) + '\n')

    spm.SentencePieceTrainer.Train(
        f'--input={temp_file} --pad_id={pad_id} --bos_id={bos_id} --eos_id={eos_id} \
        --unk_id={unk_id} --model_prefix={path}spm_{lang} --vocab_size={vocab_size} --model_type=bpe'
    )

    s = spm.SentencePieceProcessor()
    s.Load(f'{path}spm_{lang}.model')
    print(f"{lang}-dict_num: {20000}")

    return s


def tokenize(corpus, tensorlen,voca_size):  # corpus: Tokenized Sentence's List
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=voca_size,filters='',)
    tokenizer.fit_on_texts(corpus)

    tensor = tokenizer.texts_to_sequences(corpus)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post',maxlen=tensorlen)

    return tensor, tokenizer


def senten_tokenize(kor,eng, ko_model,en_model, max_len):
    kos = []
    ens = []
    for i,j in zip(kor,eng):
        i = preprocess_sentence(i)
        j = preprocess_sentence(j)
        ko = ko_model.EncodeAsIds(i)
        en = en_model.EncodeAsIds(j)
        if len(ko)>48 or len(en)>48: continue
        kos.append(ko)
        ens.append(en)
    ko_tensor = tf.keras.preprocessing.sequence.pad_sequences(kos, padding='post',maxlen=max_len)
    en_tensor = tf.keras.preprocessing.sequence.pad_sequences(ens, padding='post',maxlen=max_len)

    return ko_tensor, en_tensor


class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(LearningRateScheduler, self).__init__()

        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)

        return (self.d_model ** -0.5) * tf.math.minimum(arg1, arg2)

learning_rate = LearningRateScheduler(512)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def visualize_attention(src, tgt, enc_attns, dec_attns, dec_enc_attns):
    def draw(data, ax, x="auto", y="auto"):
        import seaborn
        seaborn.heatmap(data,
                        square=True,
                        vmin=0.0, vmax=1.0,
                        cbar=False, ax=ax,
                        xticklabels=x,
                        yticklabels=y)

    for layer in range(0, 2, 1):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print("Encoder Layer", layer + 1)
        for h in range(4):
            draw(enc_attns[layer][0, h, :len(src), :len(src)], axs[h], src, src)
        plt.show()

    for layer in range(0, 2, 1):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print("Decoder Self Layer", layer+1)
        for h in range(4):
            draw(dec_attns[layer][0, h, :len(tgt), :len(tgt)], axs[h], tgt, tgt)
        plt.show()

        print("Decoder Src Layer", layer+1)
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        for h in range(4):
            draw(dec_enc_attns[layer][0, h, :len(tgt), :len(src)], axs[h], src, tgt)
        plt.show()

# 번역 생성 함수
mecab = Mecab()
START_TOKEN = 2
END_TOKEN = 3

def evaluate(sentence, model, src_tokenizer, tgt_tokenizer, enc_len=50, dec_len=50):
    sentence = preprocess_sentence(sentence)
    try:
        pieces = src_tokenizer.encode_as_pieces(sentence)
        tokens = src_tokenizer.encode_as_ids(sentence)

        _input = tf.keras.preprocessing.sequence.pad_sequences([tokens],
                                                            maxlen=enc_len,
                                                            padding='post')

        ids = []

        output = tf.expand_dims([tgt_tokenizer.bos_id()], 0)
        for i in range(dec_len):
            enc_padding_mask, combined_mask, dec_padding_mask = \
            generate_masks(_input, output)

            predictions, enc_attns, dec_attns, dec_enc_attns =\
            model(_input,
                output,
                enc_padding_mask,
                combined_mask,
                dec_padding_mask)

            predicted_id = \
            tf.argmax(tf.math.softmax(predictions, axis=-1)[0, -1]).numpy().item()

            if tgt_tokenizer.eos_id() == predicted_id:
                result = tgt_tokenizer.decode_ids(ids)
                return pieces, result, enc_attns, dec_attns, dec_enc_attns

            ids.append(predicted_id)
            output = tf.concat([output, tf.expand_dims([predicted_id], 0)], axis=-1)

        result = tgt_tokenizer.decode_ids(ids)

        return pieces, result, enc_attns, dec_attns, dec_enc_attns
    except:
        pieces = mecab.morphs(sentence)
        tokens = src_tokenizer.texts_to_sequences([pieces])

        _input = tf.keras.preprocessing.sequence.pad_sequences(tokens,
                                                            maxlen=enc_len,
                                                            padding='post')

        ids = []

        output = tf.expand_dims([START_TOKEN], 0)
        for i in range(dec_len):
            enc_padding_mask, combined_mask, dec_padding_mask = generate_masks(_input, output)

            predictions, enc_attns, dec_attns, dec_enc_attns = \
                model(_input, output, enc_padding_mask, combined_mask, dec_padding_mask)

            predicted_id = \
                tf.argmax(tf.math.softmax(predictions, axis=-1)[0,-1]).numpy().item()

            if predicted_id == END_TOKEN:
                result = tgt_tokenizer.sequences_to_texts([ids])
                return pieces, result, enc_attns, dec_attns, dec_enc_attns

            ids.append(predicted_id)
            output = tf.concat([output, tf.expand_dims([predicted_id], 0)], axis=-1)
        
        result = tgt_tokenizer.sequences_to_texts([ids])

        return pieces, result, enc_attns, dec_attns, dec_enc_attns



# 번역 생성 및 Attention 시각화 결합

def translate(sentence, model, src_tokenizer, tgt_tokenizer, plot_attention=False):
    pieces, result, enc_attns, dec_attns, dec_enc_attns = \
    evaluate(sentence, model, src_tokenizer, tgt_tokenizer)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    if plot_attention:
        visualize_attention(pieces, result.split(), enc_attns, dec_attns, dec_enc_attns)
    return result



from tqdm import tqdm

class Transformer(tf.keras.Model):
    def __init__(self, n_layers,d_model,
                 n_heads,dff,src_vocab_size,
                 tgt_vocab_size, pos_len, loss_function,
                 dropout=0.2, shared=True):
        super(Transformer, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)

        self.enc_embedding = tf.keras.layers.Embedding(src_vocab_size, d_model)
        self.dec_embedding = tf.keras.layers.Embedding(tgt_vocab_size, d_model)

        self.positional = positional_encoding(pos_len,d_model)

        self.encoder = Encoder(n_layers, d_model, n_heads, dff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, dff, dropout)
        
        self.out_linear = tf.keras.layers.Dense(tgt_vocab_size)
        self.dropout = tf.keras.layers.SpatialDropout1D(dropout/2)

        self.shared = shared
        self.history = {'loss':[],'val_loss':[], 'attention':[]}

        self.loss_function = loss_function

        self.x_len = None
        self.y_len = None

        if shared: self.out_linear.set_weights(tf.transpose(self.dec_embedding.weights)) # 이런 생각을 한다는 것이 매우 놀랍다.

    
    def embedding(self, emb, x,training=False):
        # share?
        seq_len = x.shape[1]
        out = emb(x)
        if self.shared: out *= tf.math.sqrt(self.d_model)

        out += self.positional[np.newaxis, ...][:, :seq_len, :]
        out = self.dropout(out, training=training)

        return out

    
    def call(self, enc_in, dec_in, enc_mask, causality_mask, dec_mask, training=False):
        # 1 embedding
        if self.x_len is None: self.x_len = enc_in.shape[1]
        if self.y_len is None: self.y_len = dec_in.shape[1]
        enc = self.embedding(self.enc_embedding, enc_in, training)
        dec = self.embedding(self.dec_embedding, dec_in, training)

        # 2 encoder, decoder
        enc_out, enc_attns = self.encoder(enc, enc_mask, training)

        dec_out, dec_attns, dec_enc_attns = self.decoder(dec, enc_out, causality_mask, dec_mask, training)

        logits = self.out_linear(dec_out)

        return logits, enc_attns, dec_attns, dec_enc_attns
    
    def generate_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def generate_causality_mask(self, src_len, tgt_len):
        mask = 1 - np.cumsum(np.eye(src_len, tgt_len), 0)
        return tf.cast(mask, tf.float32)

    def generate_masks(self, src, tgt):
        enc_mask = self.generate_padding_mask(src)
        dec_mask = self.generate_padding_mask(tgt)

        dec_enc_causality_mask = self.generate_causality_mask(tgt.shape[1], src.shape[1])
        dec_enc_mask = tf.maximum(enc_mask, dec_enc_causality_mask)

        dec_causality_mask = self.generate_causality_mask(tgt.shape[1], tgt.shape[1])
        dec_mask = tf.maximum(dec_mask, dec_causality_mask)

        return enc_mask, dec_enc_mask, dec_mask

    def translate(self, sentence, src_tokenizer, tgt_tokenizer, _print=True ,plot_attention=False):
        pieces, result, enc_attns, dec_attns, dec_enc_attns = \
        evaluate(sentence, self, src_tokenizer, tgt_tokenizer, enc_len=self.x_len, dec_len=self.y_len)

        if _print:
            print('Input: %s' % (sentence))
            print('Predicted translation: {}'.format(result))

        if plot_attention:
            visualize_attention(pieces, result.split(), enc_attns, dec_attns, dec_enc_attns)

        return result

    @tf.function()
    def train_step(self, src, tgt, optimizer):
        gold = tgt[:, 1:]
            
        enc_mask, dec_enc_mask, dec_mask = self.generate_masks(src, tgt)

        with tf.GradientTape() as tape:
            predictions, enc_attns, dec_attns, dec_enc_attns = self(src, tgt, enc_mask, dec_enc_mask, dec_mask, training=True)
            loss = self.loss_function(gold, predictions[:, :-1])

        gradients = tape.gradient(loss, self.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss, enc_attns, dec_attns, dec_enc_attns




    @tf.function()
    def eval_step(self, src,tgt):
        gold = tgt[:, 1:]

        enc_mask, dec_enc_mask, dec_mask = self.generate_masks(src,tgt)

        predictions, enc_attns, dec_attns, dec_enc_attns = self(src,tgt,enc_mask, dec_enc_mask, dec_mask)
        val_loss = self.loss_function(gold, predictions[:, :-1])
        
        
        return val_loss, enc_attns, dec_attns, dec_enc_attns


    def fit(self, epochs=20, x_train=None, y_train=None,
            x_val=None, y_val=None, BATCH_SIZE=128, 
            offset_epoch=0, translate=None, examples=None, enc_tokenizer=None, dec_tokenizer=None):
        EPOCHS = epochs

        for epoch in range(EPOCHS):
            total_loss = 0
            val_loss = 0
            idx_list = list(range(0, x_train.shape[0], BATCH_SIZE))
            random.shuffle(idx_list)
            t = tqdm(idx_list)

            for (batch, idx) in enumerate(t):
                batch_loss, enc_attns, dec_attns, dec_enc_attns = \
                self.train_step(x_train[idx:idx+BATCH_SIZE],
                                y_train[idx:idx+BATCH_SIZE],
                                optimizer)

                total_loss += batch_loss
                
                t.set_description_str('Epoch %2d' % (offset_epoch+epoch + 1))
                t.set_postfix_str('Loss %.4f' % (total_loss.numpy() / (batch + 1)))
            self.history['loss'].append(total_loss.numpy() / (batch + 1))

            if x_val is not None:
                idx_list = list(range(0, x_val.shape[0], BATCH_SIZE))[:-2]
                random.shuffle(idx_list)
                t = tqdm(idx_list)

                for (batch, idx) in enumerate(t):
                    batch_loss, enc_attns, dec_attns, dec_enc_attns = \
                        self.eval_step(x_val[idx:idx+BATCH_SIZE],
                                    y_val[idx:idx+BATCH_SIZE])
                    
                    val_loss += batch_loss
                    
                    t.set_description_str('Val_epoch %2d' % (offset_epoch+epoch + 1))
                    t.set_postfix_str('Val_loss %.4f' % (val_loss.numpy() / (batch + 1)))
                self.history['val_loss'].append(val_loss.numpy() / (batch + 1))

            if examples and translate:
                for ex in examples:
                    self.translate(ex, enc_tokenizer, dec_tokenizer)
                time.sleep(2)

        self.history['attention'].extend([enc_attns, dec_attns, dec_enc_attns])
        return self.history


class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(LearningRateScheduler, self).__init__()
        
        self.d_model = d_model
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return (self.d_model ** -0.5) * tf.math.minimum(arg1, arg2)

learning_rate = LearningRateScheduler(512)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)