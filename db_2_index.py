import argparse
import json
import sqlite3
import annoy
import os


def json_2_object(obj):
    return json.loads(obj)


def run(input, output):
    if os.path.isfile(output):
        os.remove(output)

    db_connection = sqlite3.connect(input)
    index = annoy.AnnoyIndex(128, 'euclidean')

    with db_connection:
        cur = db_connection.cursor()
        cur.execute("SELECT * FROM features")

        rows = cur.fetchall()

        for row in rows:
            idx = row[0]
            file_name = row[1]
            feature = json_2_object(row[2])

            index.add_item(idx, feature)

            print(idx, file_name, feature)

        index.build(100)
        index.save(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DB to index',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', dest='input', type=str, metavar='',
                        required=False, help='DB path', default='output.db')
    parser.add_argument('-o', '--output', dest='output', type=str, metavar='',
                        required=False, help='Index path', default='index.ann')

    args = parser.parse_args()

    print('*' * 50)
    for i in vars(args):
        print(str(i) + ' - ' + str(getattr(args, i)))

    print('*' * 50)

    run(args.input, args.output)
