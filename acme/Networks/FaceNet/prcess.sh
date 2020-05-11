conda install -c conda-forge blaze dlib=19.4

python preprocess.py --input-dir data --output-dir output/intermediate --crop-dim 180

python download_and_extract_model.py --model-dir weights

#python train_classifier.py --input-dir output/intermediate --model-path etc/20170511-185253/20170511-185253.pb --classifier-path output/classifier.pkl --num-threads 16 --num-epochs 25 --min-num-images-per-class 10 --is-train
python train_classifier.py --input-dir output/intermediate --model-path weights/20170511-185253/20170511-185253.pb --classifier-path output/classifier.pkl --num-threads 16 --num-epochs 25 --min-num-images-per-class 10 --is-train


#Eval
docker run -v $PWD:/$(basename $PWD) \
-e PYTHONPATH=$PYTHONPATH:/medium-facenet-tutorial \
-it colemurray/medium-facenet-tutorial \
python3 /medium-facenet-tutorial/medium_facenet_tutorial/train_classifier.py \
--input-dir /medium-facenet-tutorial/output/intermediate \
--model-path /medium-facenet-tutorial/etc/20170511-18253/20170511-185253.pb \
--classifier-path /medium-facenet-tutorial/output/classifier.pkl \
--num-threads 16 \
--num-epochs 5 \
--min-num-images-per-class 10