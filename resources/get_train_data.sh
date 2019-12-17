wget 'https://gitlab.com/federicozzo/attgan/raw/master/resources/train.tfrecords.zip'
unzip train.tfrecords.zip
rm -f 'train.tfrecords.zip'
mv 'train.tfrecords' resources
