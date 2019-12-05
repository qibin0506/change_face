import cv2
import numpy as np
import dlib
import math
from glob import glob

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")


def read_imgs():
    imgs = []
    for img in sorted(glob("./imgs/*")):
        imgs.append(cv2.imread(img))

    return imgs


def to_video(name, width, height):
    fps = 10
    video_writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, (width, height))

    return video_writer


def get_equation(x0, y0, x1, y1, pow_arg=1):
    k = (y1 - y0) / (pow(x1, pow_arg) - pow(x0, pow_arg))
    b = y0 - k * pow(x0, pow_arg)

    def f(x):
        return k * pow(x, pow_arg) + b

    return f


def face_detector(img):
    dets = detector(img, 1)

    # 目前仅拿一张脸
    landmarks = []
    shape = predictor(img, dets[0])
    for p in shape.parts():
        landmarks.append(np.array([p.x, p.y]))

    return dets[0], np.array(landmarks)


def get_rotate_theta(from_landmarks, to_landmarks):
    from_left_eye = from_landmarks[36]
    from_right_eye = from_landmarks[45]

    to_left_eye = to_landmarks[36]
    to_right_eye = to_landmarks[45]

    from_angle = math.atan2(from_right_eye[1] - from_left_eye[1], from_right_eye[0] - from_left_eye[0])
    to_angle = math.atan2(to_right_eye[1] - to_left_eye[1], to_right_eye[0] - to_left_eye[0])

    from_theta = -from_angle * (180 / math.pi)
    to_theta = -to_angle * (180 / math.pi)

    return to_theta - from_theta


def after_rotated(x, y, cx, cy, theta, scale):
    new_x = x - cx
    new_y = y - cy

    new_x = np.cos(theta) * new_x * scale + np.sin(theta) * new_y * scale + cx
    new_y = -np.sin(theta) * new_x * scale + np.cos(theta) * new_y * scale + cy

    return new_x, new_y


def compose_img(name, frames_per_transformer, wait_frames, *imgs):
    video_writer = to_video("{}.avi".format(name), imgs[0].shape[1], imgs[0].shape[0])

    img_count = len(imgs)
    for idx in range(img_count - 1):
        from_img = imgs[idx]
        to_img = imgs[idx + 1]

        from_width = from_img.shape[1]
        from_height = from_img.shape[0]

        to_width = to_img.shape[1]
        to_height = to_img.shape[0]

        from_face_region, from_landmarks = face_detector(from_img)
        to_face_region, to_landmarks = face_detector(to_img)

        # 第一张图最终的旋转角度
        from_theta = get_rotate_theta(from_landmarks, to_landmarks)
        # 第二张图初始的旋转角度
        to_theta = get_rotate_theta(to_landmarks, from_landmarks)

        # 两张图的旋转中心
        from_rotate_center = (from_face_region.left() + (from_face_region.right() - from_face_region.left()) / 2, from_face_region.top() + (from_face_region.bottom() - from_face_region.top()) / 2)
        to_rotate_center = (to_face_region.left() + (to_face_region.right() - to_face_region.left()) / 2, to_face_region.top() + (to_face_region.bottom() - to_face_region.top())/2)

        from_face_area = from_face_region.area()
        to_face_area = to_face_region.area()

        # 第一张图的最终缩放因子
        to_scaled = from_face_area / to_face_area
        # 第二张图的初始缩放因子
        from_scaled = to_face_area / from_face_area

        # 平移多少的基准
        to_translation_base = to_rotate_center
        from_translation_base = from_rotate_center

        equation_pow = 1 if idx % 2 == 0 else 2

        # 建立变换角度的方程
        to_theta_f = get_equation(0, to_theta, frames_per_transformer - 1, 0, equation_pow)
        from_theta_f = get_equation(0, 0, frames_per_transformer - 1, from_theta, equation_pow)

        # 建立缩放系数的角度
        to_scaled_f = get_equation(0, to_scaled, frames_per_transformer - 1, 1, equation_pow)
        from_scaled_f = get_equation(0, 1, frames_per_transformer - 1, from_scaled, equation_pow)

        for i in range(frames_per_transformer):
            # 当前时间点的旋转角度
            cur_to_theta = to_theta_f(i)
            cur_from_theta = from_theta_f(i)

            # 当前时间点的缩放因子
            cur_to_scaled = to_scaled_f(i)
            cur_from_scaled = from_scaled_f(i)

            # 生成第二张图片变换矩阵
            to_rotate_M = cv2.getRotationMatrix2D(to_rotate_center, cur_to_theta, cur_to_scaled)
            # 对第二张图片执行仿射变换
            to_dst = cv2.warpAffine(to_img, to_rotate_M, (to_width, to_height), borderMode=cv2.BORDER_REPLICATE)

            # 生成第一张图片的变换矩阵
            from_rotate_M = cv2.getRotationMatrix2D(from_rotate_center, cur_from_theta, cur_from_scaled)
            # 对第一张图片执行仿射变换
            from_dst = cv2.warpAffine(from_img, from_rotate_M, (from_width, from_height), borderMode=cv2.BORDER_REPLICATE)

            # 重新计算变换后的平移基准
            to_left_rotated = to_rotate_M[0][0] * to_translation_base[0] + to_rotate_M[0][1] * to_translation_base[1] + to_rotate_M[0][2]
            to_top_rotated = to_rotate_M[1][0] * to_translation_base[0] + to_rotate_M[1][1] * to_translation_base[1] + to_rotate_M[1][2]

            from_left_rotated = from_rotate_M[0][0] * from_translation_base[0] + from_rotate_M[0][1] * from_translation_base[1] + from_rotate_M[0][2]
            from_top_rotated = from_rotate_M[1][0] * from_translation_base[0] + from_rotate_M[1][1] * from_translation_base[1] + from_rotate_M[1][2]

            # 当前时间点的平移数
            to_left_f = get_equation(0, from_left_rotated - to_left_rotated, frames_per_transformer - 1, 0, equation_pow)
            to_top_f = get_equation(0, from_top_rotated - to_top_rotated, frames_per_transformer - 1, 0, equation_pow)

            from_left_f = get_equation(0, 0, frames_per_transformer - 1, to_left_rotated - from_left_rotated, equation_pow)
            from_top_f = get_equation(0, 0, frames_per_transformer - 1, to_top_rotated - from_top_rotated, equation_pow)

            # 生成第二张图片平移的变换矩阵
            to_translation_M = np.float32(
                [
                    [1, 0, to_left_f(i)],
                    [0, 1, to_top_f(i)]
                ]
            )

            # 对第二张图片执行平移变换
            to_dst = cv2.warpAffine(to_dst, to_translation_M, (to_width, to_height), borderMode=cv2.BORDER_REPLICATE)

            # 生成第一张图片平移的变换矩阵
            from_translation_M = np.float32(
                [
                    [1, 0, from_left_f(i)],
                    [0, 1, from_top_f(i)]
                ]
            )

            # 对第一张图片执行平移变换
            from_dst = cv2.warpAffine(from_dst, from_translation_M, (from_width, from_height), borderMode=cv2.BORDER_REPLICATE)

            # 将两张图片合成到一张，并写入视频帧
            new_img = cv2.addWeighted(from_dst, 1 - ((i + 1) / frames_per_transformer), to_dst, (i + 1) / frames_per_transformer, 0)
            video_writer.write(new_img)

        # 一个迭代完成，迭代n次写入第二张图片
        for _ in range(wait_frames):
            video_writer.write(to_img)

    video_writer.release()


imgs = read_imgs()
compose_img("./affine_result", 15, 5, *imgs)
