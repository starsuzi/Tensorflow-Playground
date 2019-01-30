import tensorflow as tf

hello = tf.constant('Hello, TesorFlow!')
print(hello)

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a,b)

print(c)

#위에서 변수와 수식들을 정의했지만, 실행이 정의한 시점에서 실행되는 것은 아니다.
#다음처럼 session 객체와 run 메소드를 사용할 때 계산이 된다.
#따라서 모델을 구성하는 것과 실행하는 것을 분리하여 프로그램을 깔끔하게 작성할 수 있다. 

#그래프를 실행할 세션 구성
sess = tf.Session()

#sess.run: 설정한 텐서 그래프, 변수, 수식을 실행
print(sess.run(hello))
print(sess.run([a,b,c]))

sess.close()