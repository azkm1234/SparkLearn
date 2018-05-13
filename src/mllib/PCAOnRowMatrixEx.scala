package mllib

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, Vectors}
import org.apache.spark.{SparkConf, SparkContext}

object PCAOnRowMatrixEx {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setMaster("local[2]").setAppName("PCAOnRowMatrix"))
    val data = Array(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0))

    val rows = sc.parallelize(data)
    val mat = new RowMatrix(rows)
    val pc: Matrix = mat.computePrincipalComponents(3)
    println(pc)
    val projected: RowMatrix = mat.multiply(pc)

    val collect = projected.rows.collect()
    println("Projected Row Matrix of principal component:")
    collect.foreach { vector => println(vector) }

  }
}
