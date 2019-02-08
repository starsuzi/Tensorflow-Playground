import tensorflow as tf
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np

#matplotlib에서 한글을 표시하기 위한 설정
#font_name = matplotlib.font_manager.FontProperties(
#                fname="/Library/Fonts/NanumGothic.otf"  # 한글 폰트 위치를 넣어주세요
#            ).get_name()
#matplotlib.rc('font', family=font_name)

#단어 벡터를 분석해볼 임의의 문장들
sentences = [
    'I like dogs',
    'I like cats',
    'I like animals',
    'dogs cats animals',
    'boyfriend like cats dogs',
    'cats like fish milk',
    'dogs hate fish like milk',
    'dogs cats like eyes ',
    'I like boyfriend',
    'boyfriend hate I',
    'boyfriend I movie book music like',
    'I game comic drama',
    'cats dogs hate',
    'dogs cats like'
]

#문장을 전부 합친 수 공백으로 단어들을 나누고 고유한 단어들로 리스트 생성
word_sequence = ' '.join(sentences).split()
#print(word_sequence)
#set은 중복 허용 하지 않음 -> 단어의 종류를 중복 없이 볼 수 있다.
word_list = ' '.join(sentences).split()
word_list = list(set(word_list))
#문자열로 분석하는 것보다, 숫자로 분석하는 것이 훨씬 용이하므로
#리스트에서 문자들의 인덱스를 뽑아서 사용하기 위해
#이를 표현하기 위한 연관 배열과 단어리스트에서 단어를 참조할 수 있는 인덱스 배열 생성
word_dict = {w: i for i, w in enumerate(word_list)}
#print(word_dict)

# 윈도우 사이즈를 1 로 하는 skip-gram 모델을 만듭니다.
# 예) 나 게임 만화 애니 좋다
#   -> ([나, 만화], 게임), ([게임, 애니], 만화), ([만화, 좋다], 애니)
#   -> (게임, 나), (게임, 만화), (만화, 게임), (만화, 애니), (애니, 만화), (애니, 좋다)
skip_grams = []

for i in range(1, len(word_sequence)-1):
    #(context, target) : ([target index -1, target index + 1], target)
    #스킵그램을 만든 후, 저장은 단어의 고유 번호 (index)로 저장
    target = word_dict[word_sequence[i]]
    context = word_dict[word_sequence[i-1]], word_dict[word_sequence[i+1]]

    #(target, context[0]), (target, context[1])...
    for w in context:
        skip_grams.append([target, w])

#print(skip_grams)

#skip-gram 데이터에서 무작위로 데이터를 뽑아 입력값과 출력값의 배치 데이터를 생성하는 함수
def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace = False)

    for i in random_index:
        random_inputs.append(data[i][0]) #target
        random_labels.append(data[i][1]) #context

    return random_inputs, random_labels

####
#option
####
#학습을 반복할 횟수
training_epoch = 300
#학습률
learning_rate = 0.1
#한번에 학습할 데이터의 크기
batch_size = 20
#단어 벡터를 구성할 임베딩 차원의 크기
#이 예제에서는 x, y 그래프로 표현하기 쉽게 2개의 값만 출력
embedding_size = 2
#word2vec 모델을 학습시키기 위한 nce_loss 함수에서 사용하기 위한 샘플링 크기
#batch_size보다 작아야함
num_sampled = 15
#총 단어 개수
voc_size = len(word_list)

####
#신경망 모델 구성
####
inputs = tf.placeholder(tf.int32, shape = [batch_size])
#tf.nn.nce_loss를 사용하려면 출력값을 [batch_size, 1]로 구성해야 함
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

#word2vec 모델의 결과 값인 임베딩 벡터를 저장할 변수
#총 단어 개수와 임베딩 개수를 크기로 하는 두 개의 차원을 가짐
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
#임베딩 벡터의 차원에서 학습할 입력값에 대한 행들을 뽑아옴
# 예) embeddings     inputs    selected
#    [[1, 2, 3]  -> [2, 3] -> [[2, 3, 4]
#     [2, 3, 4]                [3, 4, 5]]
#     [3, 4, 5]
#     [4, 5, 6]]
selected_embd = tf.nn.embedding_lookup(embeddings, inputs)

#nce_loss 함수에서 사용할 변수들 정의
nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

#nce_loss 함수를 직접 구현하려면 매우 복잡하지만,
#함수를 텐서플로우가 제공하므로 그냥 tf.nn.nce_loss 함수를 사용하면 된다
loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embd, num_sampled, voc_size)
)

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

####
#신경망 모델 학습
####
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(1, training_epoch+1):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)
        
        print(batch_inputs)
        print(batch_labels)

        _, loss_val = sess.run([train_op, loss],
                                feed_dict = {inputs: batch_inputs, 
                                             labels: batch_labels})

        if step % 10 == 0:
            print('loss at step ', step, ': ', loss_val)

    #matplot으로 출력하여 시각적으로 확인해보기 위해
    #임베딩 벡터의 결과 값을 계산하여 저장
    #with 구문안에서는 sess.run대신 간단히 eval()함수를 사용 가능
    trained_embeddings = embeddings.eval()

####
#임베딩된 word2vec 결과 확인
#결과는 해당 단어들이 얼마나 다른 단어와 인접해 있는지를 보여줌
####
#for i, label in enumerate(word_list):
    #print(trained_embeddings[i])
    #x,y = trained_embeddings[i]
    #print(x,y)