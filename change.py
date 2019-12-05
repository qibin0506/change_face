import cv2
import numpy as np
import dlib
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

    # only support one face
    landmarks = []
    shape = predictor(img, dets[0])
    for p in shape.parts():
        landmarks.append(np.array([p.x, p.y]))

    return dets[0], np.array(landmarks)


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

        equation_pow = 1 if idx % 2 == 0 else 2

        from_face_region, from_landmarks = face_detector(from_img)
        to_face_region, to_landmarks = face_detector(to_img)

        # 建立关键点的方程
        homography_equation = get_equation(0, from_landmarks, frames_per_transformer - 1, to_landmarks, equation_pow)

        for i in range(frames_per_transformer):
            # 计算第一张图的最佳变换矩阵
            from_H, _ = cv2.findHomography(from_landmarks, homography_equation(i))
            # 计算第二张图的最佳变换矩阵
            to_H, _ = cv2.findHomography(to_landmarks, homography_equation(i))

            # 对第一张图作透视变换
            from_dst = cv2.warpPerspective(from_img, from_H, (from_width, from_height), borderMode=cv2.BORDER_REPLICATE)
            # 对第二张图作透视变换
            to_dst = cv2.warpPerspective(to_img, to_H, (to_width, to_height), borderMode=cv2.BORDER_REPLICATE)

            new_img = cv2.addWeighted(from_dst, 1 - ((i + 1) / frames_per_transformer), to_dst, (i + 1) / frames_per_transformer, 0)
            video_writer.write(new_img)

        for _ in range(wait_frames):
            video_writer.write(to_img)

    video_writer.release()


imgs = read_imgs()
compose_img("./result", 15, 5, *imgs)
