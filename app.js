const MODEL_PATH = location.href.includes("github.io")
    ? "/the-classification-of-corn-diseases/my_model/"
    : "./my_model/";

let model;
let webcam;
let currentImage = null;

window.addEventListener("DOMContentLoaded", init);

async function init() {
    const modelURL = MODEL_PATH + "model.json";
    const metadataURL = MODEL_PATH + "metadata.json";

    model = await tmImage.load(modelURL, metadataURL);

    webcam = new tmImage.Webcam(340, 340, true);
    await webcam.setup({ facingMode: "environment" });
    await webcam.play();

    document.getElementById("webcam-container").appendChild(webcam.canvas);
    requestAnimationFrame(loop);
}

async function loop() {
    if (webcam) webcam.update();
    requestAnimationFrame(loop);
}

function capture() {
    const canvas = document.createElement("canvas");
    canvas.width = webcam.canvas.width;
    canvas.height = webcam.canvas.height;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(webcam.canvas, 0, 0);

    currentImage = canvas;

    const container = document.getElementById("webcam-container");
    container.innerHTML = "";
    container.appendChild(canvas);
}

async function classifyImage() {
    if (!currentImage) {
        alert("กรุณาถ่ายภาพก่อน");
        return;
    }

    const prediction = await model.predict(currentImage);
    showResult(prediction);
}

async function resetCamera() {
    document.getElementById("webcam-container").innerHTML = "";
    await webcam.play();
    document.getElementById("webcam-container").appendChild(webcam.canvas);
    currentImage = null;
    document.getElementById("result").innerHTML = "";
}

document.getElementById("uploadInput").addEventListener("change", function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = function() {
        currentImage = img;
        const container = document.getElementById("webcam-container");
        container.innerHTML = "";
        container.appendChild(img);
    };
});

async function classifyUpload() {
    if (!currentImage) {
        alert("กรุณาเลือกรูปก่อน");
        return;
    }

    const prediction = await model.predict(currentImage);
    showResult(prediction);
}

function showResult(predictions) {
    const best = predictions.reduce((a, b) =>
        a.probability > b.probability ? a : b
    );

    const percent = (best.probability * 100).toFixed(1);

    document.getElementById("result").innerHTML = `
        <h2>${best.className}</h2>
        <div class="progress-bar">
            <div class="progress-fill" style="width:${percent}%"></div>
        </div>
        <p>Confidence: ${percent}%</p>
    `;
}
