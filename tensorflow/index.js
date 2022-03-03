

async function app() {

    console.log('Loading mobilenet..');

    // Load the model.
    const model = await tf.loadLayersModel('http://localhost:8003/model.json');
    console.log('Successfully loaded model');

    // Make a prediction through the model on our image.
    const imgEl = document.getElementById('img');
    const tensor = tf.browser.fromPixels(imgEl);
    const resized = tf.image.resizeBilinear(tensor, [150, 150]).toFloat()
    const offset = tf.scalar(255.0);
    const normalized = tf.scalar(1.0).sub(resized.div(offset));
    const tensor_expand = normalized.expandDims(0)
    var res = await model.predict(tensor_expand);
    console.log('Done')
    console.log(res.arraySync()[0]);
    const pred_value = res.arraySync()[0]

    if (pred_value > 0.5) {
        document.getElementById('prediction').innerHTML += '<br>NO (' + pred_value + ')';
    }
    else {
        document.getElementById('prediction').innerHTML += '<br>YES (' + pred_value + ')';
    }
}
const img = document.getElementById('img')
img.onload = function () {

    app();
}