<h2> Edge detection kernels in the first Conv2D layer of an object detection model</h2>

Main goal of this lab assignment is to check the hypothesis that the first convolutional layer in an object detection model performs edge detection among other feature extraction tasks.

Our tasks:
1. Analyze learned kernel weights of the first convolutional layer in the [SSD head detection](https://github.com/AVAuco/ssd_head_keras) model. Carry on a convolution operation on an example image using the retrieved kernel weights sand analyze the results.
2. Run a simple edge detection model based on Sobel filter using TensorFlow functional API. Test it on few example images to see the basics of edge detection.

<table>
  <tr>
        <th width="450px">Fig 1. Kernel weights of the first convolutional layer in the SSD head detection model</th>
        <th>Fig 2. Result of Sobel filtering without and with threshold (Sobel binary filtering)</th>
  </tr>
  <tr>
    <td>
        <img src="Lab_2_1.png" width="400px"/>
    </td>
    <td>
        <img src="Lab_2_2.png" width="550px">
    </td>
  </tr>
</table>
