import tensorflow as tf 
#TensorFlow 를 사용하기위해 import

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)
#MNIST 데이터를 다운로드 한다.

x = tf.placeholder(tf.float32, [None, 784])
#TensorFlow에게 계산을 하도록 명령할 때 입력할 값
#각 이미지들은 784차원의 벡터로 단조화
#[None, 784] 형태의 부정소숫점으로 이루어진 2차원 텐서로 표현

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#가중치와 편향값을 추가적인 입력으로 다룸
#Variable은 TensorFlow의 상호작용하는 작업 그래프들간에 유지되는 변경 가능한 텐서
#tf.Variable 을 주어서 Variable의 초기값을 만듦
#W 와 b 를 0으로 채워진 텐서들로 초기화

y = tf.nn.softmax(tf.matmul(x, W) + b)
#tf.matmul(x, W) 표현식으로 xx 와 WW를 곱함
#그 다음 b를 더하고, 마지막으로 tf.nn.softmax 를 적용

y_ = tf.placeholder(tf.float32, [None, 10])
#교차 엔트로피를 구현하기 위해 우리는 우선적으로 정답을 입력하기 위한 새 placeholder를 추가

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#교차 엔트로피 −∑y′log(y)−∑y′log⁡(y) 를 구현
#tf.log는 y의 각 원소의 로그값을 계산
#y_ 의 각 원소들에, 각각에 해당되는 tf.log(y)를 곱함
#tf.reduce_sum은 텐서의 모든 원소를 더함

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#TensorFlow에게 학습도를 0.01로 준 경사 하강법(gradient descent) 알고리즘을 이용하여 교차 엔트로피를 최소화하도록 명령
#TensorFlow가 여기서 실제로 (뒤에서) 하는 일은 역전파 및 경사 하강법을 구현한 작업을 당신의 그래프에 추가

init = tf.initialize_all_variables()
#변수들을 초기화하는 작업을 추가

sess = tf.Session()
sess.run(init)
#세션에서 모델을 시작하고 변수들을 초기화하는 작업을 실행

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#학습을 1000번함
#각 반복 단계마다, 학습 세트로부터 100개의 무작위 데이터들의 일괄 처리(batch)들을 가져옴
#placeholders를 대체하기 위한 일괄 처리 데이터에 train_step 피딩을 실행

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#tf.argmax(y,1) 는 진짜 라벨이 tf.argmax(y_,1) 일때 우리 모델이 각 입력에 대하여 가장 정확하다고 생각하는 라벨
#tf.equal 을 이용해 예측이 실제와 맞았는지 확인할 수 있음

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#이 결과는 부울 리스트를 줌
#얼마나 많은 비율로 맞았는지 확인하려면, 부정소숫점으로 캐스팅한 후 평균값을 구하면 됨

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#테스트 데이터를 대상으로 정확도를 확인