

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from time import time

df = spark.read.csv(path='./train.csv', inferSchema=True, header=True)
df = df.withColumn('foreclosure_status', df['foreclosure_status'].cast('double'))

## Let's stratify the data since we have a small amount of Foreclosures
positive_count = df.filter(df['foreclosure_status'] == 1.0).count()
data_size = df.count()
strat_data = df.sampleBy('foreclosure_status', fractions={0: float(positive_count)/ data_size, 1: 1.0})
print strat_data.groupby('foreclosure_status').count().toPandas()

splitSeed = 12345
train_data, test_data = strat_data.randomSplit([0.8, 0.2], splitSeed)


feature_cols = df.drop('foreclosure_status').drop('id').columns
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

lr = LogisticRegression(labelCol='foreclosure_status', featuresCol='features')

pipeline = Pipeline(stages=[assembler, lr])


paramGrid = ParamGridBuilder() \
    .addGrid(lr.maxIter, [1, 10, 100]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()
    

    
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol='foreclosure_status', predictionCol='prediction'),
                          numFolds=3)

time_s = time()
cv_model = crossval.fit(train_data)
time_e = time()

print 'Total training time: %f' % (time_e - time_s)
```

    Total training time: 405.023449



```python
def print_metrics(predictions_and_labels):
    metrics = MulticlassMetrics(predictions_and_labels)
    print 'Precision of True ', metrics.precision(1)
    print 'Precision of False', metrics.precision(0)
    print 'Recall of True    ', metrics.recall(1)
    print 'Recall of False   ', metrics.recall(0)
    print 'F-1 Score         ', metrics.fMeasure()
    print 'Confusion Matrix\n', metrics.confusionMatrix().toArray()
    
predictions = cv_model.transform(test_data)
accuracy = cv_model.getEvaluator().evaluate(predictions)
print 'F1 Accuracy: %f' % accuracy

predictions_and_labels = predictions.select("prediction", "foreclosure_status").rdd \
.map(lambda r: (float(r[0]), float(r[1])))

print_metrics(predictions_and_labels)
```

    F1 Accuracy: 0.765514
    Precision of True  0.757575757576
    Precision of False 0.772727272727
    Recall of True     0.714285714286
    Recall of False    0.809523809524
    F-1 Score          0.766233766234
    Confusion Matrix


    /usr/local/Cellar/apache-spark/2.0.1/libexec/python/pyspark/mllib/evaluation.py:262: UserWarning: Deprecated in 2.0.0. Use accuracy.
      warnings.warn("Deprecated in 2.0.0. Use accuracy.")


    [[ 34.   8.]
     [ 10.  25.]]



```python

```
