docker run -v $PWD:/medium-facenet-tutorial \
-e PYTHONPATH=$PYTHONPATH:/medium-facenet-tutorial \
-it colemurray/medium-facenet-tutorial python3 /medium-facenet-tutorial/medium_facenet_tutorial/preprocess.py \
--input-dir /medium-facenet-tutorial/data \
--output-dir /medium-facenet-tutorial/output/intermediate \
--crop-dim 180