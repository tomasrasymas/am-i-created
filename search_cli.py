import argparse
import sqlite3
import annoy
import os
import photos_2_db
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def run(database_path, index_path):
    connection = sqlite3.connect(database_path)
    index = annoy.AnnoyIndex(128, 'euclidean')
    index.load(index_path)

    image_loader = photos_2_db.ImageLoader()
    face_detector = photos_2_db.FaceDetector()
    landmarks_predictor = photos_2_db.LandmarksPredictor()
    face_recognizer = photos_2_db.FaceRecognizer()

    while True:
        input_image = input('Image path >>> ')

        if not os.path.isfile(index_path):
            print('File %s not exists.' % index_path)
            continue

        images = image_loader(file_path=input_image, width=200, height=200)
        faces = face_detector(images=images)
        landmarks = landmarks_predictor(images=images, faces=faces)
        features = face_recognizer(images=images, landmarks=landmarks)

        for feature in features[0]:
            f = list(feature)
            nearest = index.get_nns_by_vector(f, 3, search_k=-1, include_distances=True)

            for idx, distance in zip(nearest[0], nearest[1]):
                cur = connection.cursor()
                cur.execute('SELECT * FROM features where id=%s' % idx)

                rows = cur.fetchall()

                for row in rows:
                    im = Image.open(row[1])
                    im.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search similar images in index',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-db', '--database', dest='database_path', type=str, metavar='',
                        required=False, help='DB path', default='output.db')
    parser.add_argument('-idx', '--index', dest='index_path', type=str, metavar='',
                        required=False, help='Index path', default='index.ann')

    args = parser.parse_args()

    print('*' * 50)
    for i in vars(args):
        print(str(i) + ' - ' + str(getattr(args, i)))

    print('*' * 50)

    run(args.database_path, args.index_path)
