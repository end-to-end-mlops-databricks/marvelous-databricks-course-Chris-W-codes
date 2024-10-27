# Databricks notebook source
import unittest
from pyspark.sql import SparkSession
from pyspark.sql import Row

# COMMAND ----------

# MAGIC %run  "/Users/chris.wroe@fabledata.com/marvelous-databricks-course-Chris-W-codes/Pre_processing"

# COMMAND ----------

class TestPreProcess(unittest.TestCase):

    def setUpClass(cls):
    # Initialize a Spark session for testing
        cls.spark = SparkSession.builder.appName("TestApp").getOrCreate()
        cls.preprocessor = PreProcess()
    
    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_append_headers(self):
    # Test that the output dataframe has the right shape
        data = [
            Row(0, "123", "2023-10-01", "NO_QUERY", "user1", "sample text 1"),
            Row(4, "124", "2023-10-02", "NO_QUERY", "user2", "sample text 2")]
        sample_df = self.spark.createDataFrame(data)
        result_df = self.preprocessor.append_headers(sample_df)
        self.assertEqual(len(result_df.columns), 6)
        #Check the dataframe has the correct column headers
        expected_columns = ["target", "id", "date", "flag", "user", "text"]
        self.assertEqual(result_df.columns, expected_columns)
         #Check the dataframe still has 2 rows
        self.assertEqual(result_df.count(), 2) 

    
    

# COMMAND ----------

test_case = TestPreProcess()
test_case.setUpClass()

# COMMAND ----------

try:
    test_case.test_append_headers()
    print("test_append_headers passed!")
except AssertionError as e:
    print("test_append_headers failed:", )
