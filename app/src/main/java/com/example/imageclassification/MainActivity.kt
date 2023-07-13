package com.example.imageclassification

import android.content.Intent
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.ContactsContract.RawContacts.Data
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.Nullable
import androidx.annotation.RequiresApi
import com.example.imageclassification.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private lateinit var camera: Button
    private lateinit var gallery: Button
    private lateinit var predict: Button
    private lateinit var result: TextView
    private lateinit var imageView: ImageView
    lateinit var image:Bitmap

    @RequiresApi(Build.VERSION_CODES.M)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        camera = findViewById<Button>(R.id.button)
        gallery = findViewById<Button>(R.id.button2)
        predict = findViewById<Button>(R.id.predict)
        result = findViewById<TextView>(R.id.result)
        imageView = findViewById<ImageView>(R.id.imageView)

        var labels = application.assets.open("labels.txt").bufferedReader().readLines()

        camera.setOnClickListener {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent, 3)
            } else {
                requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)
            }
        }
        gallery.setOnClickListener {
            val cameraIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(cameraIntent, 1)
        }

        predict.setOnClickListener {
          if(::image.isInitialized && image != null) {
              var tensorImage = TensorImage(DataType.UINT8)
              tensorImage.load(image)

              // for resizing the image compatible to the model

              var imageProcessor = ImageProcessor.Builder()
                  .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                  .build()

              tensorImage = imageProcessor.process(tensorImage)

              val model = MobilenetV110224Quant.newInstance(applicationContext)

              // Creates inputs for reference.
              val inputFeature0 =
                  TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
              inputFeature0.loadBuffer(tensorImage.buffer)

              // Runs model inference and gets result.
              val outputs = model.process(inputFeature0)
              val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

              var maxindex = 0

              outputFeature0.forEachIndexed { index: Int, fl: Float ->
                  if (outputFeature0[maxindex] < fl) {
                      maxindex = index
                  }
              }

              result.text = labels[maxindex]

              // Releases model resources if no longer used.
              model.close()
          }

        }

    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, @Nullable data: Intent?) {
        if (resultCode == RESULT_OK) {
            if (requestCode == 3) {
                image = data?.extras?.get("data") as Bitmap
                //val dimension = Math.min(image.width, image.height)
                //val thumbnail = ThumbnailUtils.extractThumbnail(image, dimension, dimension)
                imageView.setImageBitmap(image)

                //val scaledImage = Bitmap.createScaledBitmap(thumbnail, imageSize, imageSize, false)
                //classifyImage(image)
            } else {
                val uri = data?.data
                try {
                    //uniform resource indicator
                    uri?.let {
                        image = MediaStore.Images.Media.getBitmap(contentResolver, it)
                    }
                } catch (e: IOException) {
                    e.printStackTrace()
                }
                image?.let {
                    imageView.setImageBitmap(it)
                    //val scaledImage = Bitmap.createScaledBitmap(it, imageSize, imageSize, false)
                    //classifyImage(image)
                }
            }
        }
        super.onActivityResult(requestCode, resultCode, data)
    }
}