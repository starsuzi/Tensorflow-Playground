import tensorflow as tf

#placeholder란? 계산을 실행할 때 입력값을 받는 변수로 사용
#None은 크기가 정해지지 않았음을 의미
X = tf.placeholder(tf.float32, [None, 3])
print(X)

#두번째 차원의 요소의 개수는 3개
x_data = [[1,2,3], [4,5,6]]

#tf.Variable: 그래프를 계산하면서 최적화 할 변수들 -> 이 값들이 신경망을 좌우하는 값들
#tf.random_normal: 각 변수들의 초기값을 정규분포 랜덤 값으로 초기화
W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

#입력값과 변수들을 계산할 수 있는 수식을 작성
#tf.matmul처럼 mat*로 되어 있는 함수로 행렬 계산을 수행
expr = tf.matmul(X,W) + b

sess = tf.Session()
#위에서 설정한 Variable들의 값들을 초기화하기 위해
#처음에 tf.global_variables_initializer를 한 번 실행해야 함
sess.run(tf.global_variables_initializer())

print("===x_data===")
print(x_data)
print("===W===")
print(sess.run(W))
print("===b===")
print(sess.run(b))
print("===expr===")
#expr 수식에는 X라는 입력값이 필요
#따라서 expr 실행시에는 이 변수에 대한 실제 입력값을 다음처럼 넣어줘야함
print(sess.run(expr, feed_dict= {X:x_data}))

sess.close()