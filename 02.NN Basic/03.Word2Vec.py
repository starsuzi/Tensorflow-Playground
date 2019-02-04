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
    'I like dogs'
    'I like cats'
    'I like animals'
    'dogs cats animals'
    'boyfriend cats dogs like'
    'cats fish milk like'
    'dogs fish hate milk like'
    'dogs cats eyes like'
    'I like boyfriend'
    'boyfriend hate I'
    'boyfriend I movie book music like'
    'I game comic drama'
    'cats dogs hate'
    'dogs cats like'
]

#문장을 전부 합친 수 공백으로 단어들을 나누고 고유한 단어들로 리스트 생성
