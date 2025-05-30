const videoElement = document.getElementById("cam_input");
const canvasElement = document.getElementById("canvas_output");
const canvasRoi = document.getElementById("canvas_roi");
const canvasCtx = canvasElement.getContext("2d");
const roiCtx = canvasRoi.getContext("2d");

const drawingUtils = window;
const emotions = ["Angry", "Happy", "Sad", "Surprise"];
var tfliteModel;

async function start() {
  await tf
    .loadLayersModel("./static/model/uint8/model.json")
    .then((loadedModel) => {
      console.log("loadedModel", loadedModel);
      tfliteModel = loadedModel;
    });
}

start();

function openCvReady() {
  cv["onRuntimeInitialized"] = () => {
    function onResults(results) {
      try {
        // Draw the overlays.
        canvasCtx.save();
        roiCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        roiCtx.clearRect(0, 0, canvasRoi.width, canvasRoi.height);
        canvasCtx.drawImage(
          results.image,
          0,
          0,
          canvasElement.width,
          canvasElement.height
        );
        if (results.detections.length > 0) {
          drawingUtils.drawRectangle(
            canvasCtx,
            results.detections[0].boundingBox,
            { color: "blue", lineWidth: 4, fillColor: "#00000000" }
          );
          let width =
            results.detections[0].boundingBox.width * canvasElement.width;
          let height =
            results.detections[0].boundingBox.height * canvasElement.height;
          let sx =
            results.detections[0].boundingBox.xCenter * canvasElement.width -
            width / 2;
          let sy =
            results.detections[0].boundingBox.yCenter * canvasElement.height -
            height / 2;
          let center = sx + width / 2;

          let imgData = canvasCtx.getImageData(
            0,
            0,
            canvasElement.width,
            canvasElement.height
          );

          // Tạo canvas tạm thời để xử lý ROI
          let tempCanvas = document.createElement("canvas");
          tempCanvas.width = width;
          tempCanvas.height = height;
          let tempCtx = tempCanvas.getContext("2d");

          // Vẽ ROI vào canvas tạm
          tempCtx.drawImage(
            canvasElement,
            sx,
            sy,
            width,
            height,
            0,
            0,
            width,
            height
          );

          // Lấy dữ liệu ảnh từ canvas tạm
          let roiData = tempCtx.getImageData(0, 0, width, height);

          // Chuyển đổi sang grayscale
          for (let i = 0; i < roiData.data.length; i += 4) {
            let avg =
              (roiData.data[i] + roiData.data[i + 1] + roiData.data[i + 2]) / 3;
            roiData.data[i] = avg; // R
            roiData.data[i + 1] = avg; // G
            roiData.data[i + 2] = avg; // B
          }

          // Hiển thị ảnh grayscale
          roiCtx.putImageData(roiData, 0, 0);

          //issue are image is not grayscale, predict input is wrong
          const outputTensor = tf.tidy(() => {
            // Transform the image data into Array pixels.
            let img = tf.browser.fromPixels(canvasRoi);

            // Resize, normalize, expand dimensions of image pixels by 0 axis.:
            img = tf.image.resizeBilinear(img, [48, 48]);
            img = tf.div(tf.expandDims(img, 0), 255);

            // Predict the emotions.
            let outputTensor = tfliteModel.predict(img).arraySync();
            return outputTensor;
          });
          // Convert to array and take prediction index with highest value
          let index = outputTensor[0].indexOf(Math.max(...outputTensor[0]));

          canvasCtx.font = "100px Arial";
          canvasCtx.fillStyle = "red";
          canvasCtx.textAlign = "center";

          canvasCtx.fillText(emotions[index], center, sy - 10);
        }
        canvasCtx.restore();
        roiCtx.restore();
      } catch (err) {
        console.log(err.message);
      }
    }

    const faceDetection = new FaceDetection({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`;
      },
    });

    faceDetection.setOptions({
      selfieMode: true,
      model: "short",
      minDetectionConfidence: 0.1,
    });

    faceDetection.onResults(onResults);

    const camera = new Camera(videoElement, {
      onFrame: async () => {
        await faceDetection.send({ image: videoElement });
      },
      width: 854,
      height: 480,
    });

    camera.start();
  };
  cv.onRuntimeInitialized();
}