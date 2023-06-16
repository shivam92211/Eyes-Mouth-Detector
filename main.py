import cv2
import dlib
import numpy as np

#Change the path of shape_predictor
predictor_path = "C:/Users/Admin/OneDrive/Documents/SKIDS/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

def calculate_bounding_box(points):
    points = np.array(points)
    x, y, w, h = cv2.boundingRect(points)
    return (x, y, x + w, y + h)

def detect_and_draw(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    landmarks = predictor(gray, faces[0])

    left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    mouth_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]

    left_eye_box = calculate_bounding_box(left_eye_points)
    right_eye_box = calculate_bounding_box(right_eye_points)
    mouth_box = calculate_bounding_box(mouth_points)

    cv2.rectangle(image, left_eye_box[:2], left_eye_box[2:], (0, 255, 0), 2)
    cv2.rectangle(image, right_eye_box[:2], right_eye_box[2:], (0, 255, 0), 2)
    cv2.rectangle(image, mouth_box[:2], mouth_box[2:], (0, 255, 0), 2)

    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Input Different images here.
image_path = "00000.png"
detect_and_draw(image_path)
