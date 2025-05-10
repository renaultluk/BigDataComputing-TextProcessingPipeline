IP=$(hostname -i)
/opt/spark/bin/spark-submit \
    --master spark://$IP:7077 \
    --num-executors 2 \
    msbd5003.py