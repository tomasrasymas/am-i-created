import argparse
import dlib
import os
import sqlite3
import json
from concurrent.futures import ThreadPoolExecutor
import cv2


def object_2_json(obj):
    return json.dumps(obj)


class ImageLoader:
    def __call__(self, file_path):
        buffer = []
        if isinstance(file_path, list):
            for f in file_path:
                tmp_img = cv2.imread(f)
                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
                buffer.append(tmp_img)
        else:
            tmp_img = cv2.imread(file_path)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            buffer.append(tmp_img)

        return buffer


class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def __call__(self, images):
        buffer = []

        for image in images:
            buffer.append(self.detector(image, upsample_num_times=0))

        return buffer


class LandmarksPredictor:
    def __init__(self):
        self.predictor = dlib.shape_predictor('data/models/shape_predictor_68_face_landmarks.dat')

    def __call__(self, images, faces):
        buffer = []
        for image, image_faces in zip(images, faces):
            tmp_buffer = []
            for face in image_faces:
                tmp_buffer.append(self.predictor(image, face))

            buffer.append(tmp_buffer)

        return buffer


class FaceRecognizer:
    def __init__(self):
        self.recognizer = dlib.face_recognition_model_v1('data/models/dlib_face_recognition_resnet_model_v1.dat')

    def __call__(self, images, landmarks):
        buffer = []

        for image, image_landmarks in zip(images, landmarks):
            tmp_buffer = []
            for landmark in image_landmarks:
                tmp_buffer.append(self.recognizer.compute_face_descriptor(image, landmark))

            buffer.append(tmp_buffer)

        return buffer


class PhotosScanner:
    def __init__(self, batch_size=50):
        self.batch_size = batch_size

    def __call__(self, root_path):
        batch = []
        for root, sub_dirs, files in os.walk(root_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []

                    batch.append(os.path.join(root, file))

        if batch:
            yield batch


class DataStorage:
    def __init__(self, db_file_name, append):
        if not append:
            if os.path.isfile(db_file_name):
                os.remove(db_file_name)

        self.connection = sqlite3.connect(db_file_name, check_same_thread=False)

        if not append:
            self.connection.execute('DROP TABLE IF EXISTS features')

            self.connection.execute('''
            CREATE TABLE features (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_path        TEXT NOT NULL,
                feature_vector    TEXT
            );
            ''')

    def __call__(self, files_path, features):
        with self.connection as connection:
            for file_path, file_features in zip(files_path, features):
                for feature in file_features:
                    connection.execute('''
                    INSERT INTO features (photo_path, feature_vector)
                    VALUES (?, ?)
                    ''', (file_path, object_2_json(list(feature))))


def run(input, output, append, batch_size, max_workers):
    photos_scanner = PhotosScanner(batch_size=batch_size)
    image_loader = ImageLoader()
    face_detector = FaceDetector()
    landmarks_predictor = LandmarksPredictor()
    face_recognizer = FaceRecognizer()
    storage = DataStorage(db_file_name=output, append=append)
    done_files_counter = 0

    def task_done(future):
        nonlocal done_files_counter
        done_files_counter += batch_size
        print('Done %s files' % done_files_counter)

    def task(files_path):
        print('Processing %s' % files_path)
        images = image_loader(file_path=files_path)
        faces = face_detector(images=images)

        landmarks = landmarks_predictor(images=images, faces=faces)
        features = face_recognizer(images=images, landmarks=landmarks)

        storage(files_path=files_path, features=features)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for files_path in photos_scanner(root_path=input):
            future = executor.submit(task, files_path)
            future.add_done_callback(task_done)

    print('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Photos path to db',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', dest='input', type=str, metavar='',
                        required=False, help='Path of single image or root photos path to scan', default='data/images')
    parser.add_argument('-o', '--output', dest='output', type=str, metavar='',
                        required=False, help='Path of DB file', default='output.db')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, metavar='',
                        required=False, help='Size of batch to process files', default=2)
    parser.add_argument('-w', '--max_workers', dest='max_workers', type=int, metavar='',
                        required=False, help='Max workers for processing', default=2)
    parser.add_argument('-a', '--append', dest='append', action='store_true',
                        required=False, help='Append to existing DB file', default=False)

    args = parser.parse_args()

    print('*' * 50)
    for i in vars(args):
        print(str(i) + ' - ' + str(getattr(args, i)))

    print('*' * 50)

    run(args.input, args.output, args.append, args.batch_size, args.max_workers)
