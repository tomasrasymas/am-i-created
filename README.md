# Am I Created

Main purpose of project `Am I Created` is to let quick search for similar faces in images.

Recently we hear a lot about generated fake faces and I raised a question `"Is my face generated?"`. To answer this question I created this toolbox that allows to index faces found in images located in some path and perform quick search for most similar ones.

## Usage
#### Features from images
```
>>> python photos_2_db.py --help

usage: photos_2_db.py [-h] [-i] [-o] [-b] [-w] [-a]

Photos path to db

optional arguments:
  -h, --help           show this help message and exit
  -i , --input         Path of single image or root photos path to scan
                       (default: data/images)
  -o , --output        Path of DB file (default: output.db)
  -b , --batch_size    Size of batch to process files (default: 15)
  -w , --max_workers   Max workers for processing (default: 7)
  -a, --append         Append to existing DB file (default: False)
```

This script scans path provided and creates sqlite database with table named `features` with columns: `id`, `photo_path` and `feature_vector`. Database stores file path and features of found faces in it. 
#### Features to index
```
>>> python db_2_index.py --help

usage: db_2_index.py [-h] [-i] [-o]

DB to index

optional arguments:
  -h, --help      show this help message and exit
  -i , --input    DB path (default: output.db)
  -o , --output   Index path (default: index.ann)
```

This script reads features database file and stores those features in different structure for quick search (`Approximate Nearest Neighbors`). I am using `Annoy` library to search for points in space that are close to a given query point. 
#### Search
```
>>> python search_cli.py --help

usage: search_cli.py [-h] [-db] [-n] [-idx]

Search similar images in index

optional arguments:
  -h, --help         show this help message and exit
  -db , --database   DB path (default: output.db)
  -n , --n_items     `n` closest items (default: 3)
  -idx , --index     Index path (default: index.ann)
```

This script accepts face image path, reads that image and performs search in indexed faces to find most similar faces.