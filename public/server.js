var http = require('http');

var hostname = '127.0.0.1';
var port = 3005;
const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');
const fs = require('fs');
const util = require('util');
const readdir = util.promisify(fs.readdir);

const IMAGE_HEIGHT = 180;
const IMAGE_WIDTH = 180;
const NUM_CLASSES = 2;
const LEARN_RATE = 0.0001;
const DENSE_UNITS = 64;

function readImage(path) {
    const imageBuffer = fs.readFileSync(path);
    const tfimage = tfnode.node.decodeImage(imageBuffer, 3);
    const smallimg = tf.image.resizeBilinear(tfimage, [IMAGE_HEIGHT, IMAGE_WIDTH])
    const floatimg = tf.cast(smallimg, 'float32');

    return floatimg;
}

async function loadImagesDir({images}, dir) {
    let file_count = 0;
    try {
        const files = await readdir(dir);
        files.forEach(function (file) {
            if (/^(?!\._).*\.((png)|(jpg)|(jpeg)|(gif))$/gi.test(file)) {
                let this_image = readImage(dir + '/' + file);
                images.push(this_image);
                file_count++;
            }
        });
    } catch (err) {
        console.error('Unable to scan directory: ' + err);
    }
    return file_count;

}

function getLabelsSubArray(image_count, classId) {
    return Array(image_count).fill(classId);
}

async function loadImages() {
    const images = [];
    let labels = [];
    let imgCount;

    imgCount = await loadImagesDir({images}, 'flower_photos/daisy');
    labels = labels.concat(getLabelsSubArray(imgCount, 1));

    imgCount = await loadImagesDir({images}, 'flower_photos/roses');
    labels = labels.concat(getLabelsSubArray(imgCount, 2));

    return {
        testImages: tf.stack(images.slice(0, 124)),
        testLabels: tf.oneHot(tf.tensor1d(labels.slice(0, 124), 'int32'), 2),
        trainImages: tf.stack(images.slice(124)),
        trainLabels: tf.oneHot(tf.tensor1d(labels.slice(124), 'int32'), 2),
    };

}

var app = http.createServer(async function (req, res) {
        const {testImages, testLabels, trainImages, trainLabels} = await loadImages();

        //const model = createModel();

        const model = await tf.loadGraphModel('model.json');
        model.summary();

        let epochBeginTime;
        let millisPerStep;
        const validationSplit = 0.15;

        model.fit(trainImages, trainLabels, {
            epochs: 5,
            batchSize: 50,
            validationSplit
        });

        const evalOutput = model.evaluate(testImages, testLabels);
        //console.log(evalOutput[0].dataSync());
        //console.log(evalOutput[1].dataSync());


        const image = readImage("rose.jpg");
        console.log(image);
        let predict_result = await model.predict(image.reshape([1, 180, 180, 3])).data();
        console.log(predict_result);


        res.setHeader('Content-Type', 'application/json');

        res.end(
            JSON.stringify({
                firstName: "John",
                lastName: "Doe"
            })
        );


    });

function createModel() {
    let model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
        filters: 8,
        kernelSize: 5,
        strides: 1,
        activation: 'relu',
    }));

    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({
        kernelInitializer: 'VarianceScaling',
        useBias: false,
        activation: 'softmax',
        units: NUM_CLASSES
    }));

    const optimizer = tf.train.adam(LEARN_RATE);
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });


    return model;
}


app.listen(port, hostname);