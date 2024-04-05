import numpy as np


# 查看 using_files/img/PyTorch2/linear/img_1.png

# loss = (Wx + b - y)^2
def compute_error_for_given_points(W, b, points):
    """
    eg:
    points = [[1, 2],
          [3, 4],
          [5, 6]]
    :param W:
    :param b:
    :param points:
    :return:
    """
    totalError = 0
    for i in range(len(points)):
        # points[i, 0] 和 points[i, 1] 用来获取第 i 个点的 x 和 y 坐标
        x = points[i][0]
        y = points[i][1]
        totalError += (W * x + b - y) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, w_current, points, learningRate):
    """
    参考图片 using_files/img/PyTorch2/linear/img.png
    批量 梯度下降算法 的更新规则:p = p - lr * J'(p)
    :param b_current:
    :param w_current:
    :param points:
    :param learningRate:
    :return:
    """
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        # loss函数对b求导
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_w = 0  # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_given_points(initial_w, initial_b, points))
          )
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_error_for_given_points(w, b, points))
          )


if __name__ == '__main__':
    run()
