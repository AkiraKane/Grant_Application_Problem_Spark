// Basic Setup
// directives for Spark Notebook
:dp com.databricks % spark-csv_2.10 % 1.3.0
%sh -wget -P /Users/Akira/Downloads http://s3.eu-central-1.amazonaws.com/dsr-data/grants/grantsPeople.csv

val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._
import org.apache.spark.sql.functions._

// This is a Big Data University Course (Scala for Data Science)

// Loading the Data
val data = sqlContext.read.
format("com.databricks.spark.csv").
option("delimiter", "\t").
option("header", "true").
option("inferSchema", "true").
load("/Users/Akira/Downloads/grantsPeople.csv")

// Re-encode some data
val researchers = data.
withColumn("phd", data("With_PHD").equalTo("Yes").cast("Int")).
withColumn("CI", data("Role").equalTo("CHIEF_INVESTIGATOR").cast("Int")).
withColumn("paperscore", data("A2")*4 + data("A")*3)

// Summarize Team Data
val grants = researchers.groupBy("Grant_Application_ID").agg(
max("Grant_Status").as("Grant_Status"),
max("Grant_Category_Code").as("Category_Code"),
max("Contract_Value_Band").as("Value_Band"),
sum("phd").as("PHDs"),
when(max(expr("paperscore * CI")).isNull, 0).otherwise(max(expr("paperscore*CI"))).as("paperscore"),
count("*").as("teamsize"),
when(sum("Number_of_Successful_Grant").isNull,0).otherwise(sum("Number_of_Successful_Grant")).as("successes"),
when(sum("Number_of_Unsuccessful_Grant").isNull,0).otherwise(sum("Number_of_Unsuccessful_Grant")).as("failures")
)

// Hnadle Categorical Features
import org.apache.spark.ml.feature.StringIndexer

val value_band_indexer = new StringIndexer().setInputCol("Value_Band").setOutputCol("Value_Index").fit(grants)
val category_indexer = new StringIndexer().setInputCol("Category_Code").setOutputCol("Category_Index").fit(grants)
val label_indexer = new StringIndexer().setInputCol("Grant_Status").setOutputCol("status").fit(grants)

// Gather the Features into a Vector
import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler().
setInputCols(Array("Value_Index","Category_Index", "PHDs", "paperscore", "teamsize", "successes","failures")).
setOutputCol("assembled")

// Setup a Classifier
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.RandomForestClassificationModel

val rf = new RandomForestClassifier().
setFeaturesCol("assembled").
setLabelCol("status").
setSeed(42)

// ExplainParams
rf.explainParams

// Create a Pipeline
import org.apache.spark.ml.Pipeline

val pipeline = new Pipeline().setStages(Array(value_band_indexer, category_indexer, label_indexer, assembler, rf))

// Create an Evaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
val auc_eval = new BinaryClassificationEvaluator().setLabelCol("status").setRawPredictionCol("rawPrediction")
auc_eval.getMetricName

// Split into Traning and Test Datasets 
val training = grants.filter("Grant_Application_ID < 6635")
val test = grants.filter("Grant_Application_ID >= 6635")

// Run and Evaluate the Pipeline
val model = pipeline.fit(training)
val pipeline_results = model.transform(test)
auc_eval.evaluate(pipeline_results)  // 0.92

// Choosing Parameters for Tuning
rf.extractParamMap

// Simple Grid Search
import org.apache.spark.ml.tuning.ParamGridBuilder

val paramGrid = new ParamGridBuilder().addGrid(rf.maxDepth, Array(10 ,30)).addGrid(rf.numTrees, Array(10,100)).build()

// Cross Validation
import org.apache.spark.ml.tuning.CrossValidator

val cv = new CrossValidator().setEstimator(pipeline).
setEvaluator(auc_eval).
setEstimatorParamMaps(paramGrid).
setNumFolds(3)

// Final Results
val cvModel = cv.fit(training)
val cv_results = cvModel.transform(test)
auc_eval.evaluate(cv_results)

// avgMetrics
cvModel.avgMetrics

// Finding the Wining Parameters
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidatorModel

implicit class BestParamMapCrossValidationModel(cvModel: CrossValidatorModel){
	def bestEstimatorParamMap: ParamMap = {
		cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics).maxBy(_._2)._1
	}
}
println(cvModel.bestEstimatorParamMap)

// Best Model
val bestPipelineModel = cvModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
bestPipelineModel.stages

// Extracting the Winning Classifier
val bestRandomForest = bestPipelineModel.stages(4).asInstanceOf[RandomForestClassificationModel]
bestRandomForest.toDebugString
// totalNumNodes
bestRandomForest.totalNumNodes
// featureImportances 
bestRandomForest.featureImportances
