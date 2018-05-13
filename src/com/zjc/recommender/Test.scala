package com.zjc.recommender

import java.util

import org.apache.spark.mllib.linalg.Vectors

/**
  * Created by root on 2016/5/7 0007.
  */
class Test {

}
object  Test{
  def main(args: Array[String]) {
    val v = Vectors.dense(1, 2, 3, 4, 0, 0, 0, 0, 1).toSparse
    println(v(5))
  }
}
