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
    'boyfriend cats dogs like',
    'cats fish milk like',
    'dogs fish hate milk like',
    'dogs cats eyes like',
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
    