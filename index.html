<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Corn Disease Classifier</title>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.7.4/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@0.8/dist/teachablemachine-image.min.js"></script>

<style>
body{
    margin:0;
    font-family:Segoe UI,sans-serif;
    background:linear-gradient(135deg,#0f3d2e,#145c43,#1f7a5a);
    display:flex;
    justify-content:center;
    align-items:center;
    height:100vh;
    color:white;
}
.container{
    width:420px;
    background:rgba(255,255,255,0.1);
    backdrop-filter:blur(15px);
    padding:30px;
    border-radius:20px;
    text-align:center;
}
video,img{
    width:100%;
    border-radius:15px;
    margin-top:15px;
}
button,input[type="file"]{
    margin:8px;
    padding:12px 20px;
    border:none;
    border-radius:30px;
    font-weight:bold;
    cursor:pointer;
}
.primary{
    background:linear-gradient(90deg,#00c853,#00e676);
    color:black;
}
.secondary{
    background:rgba(255,255,255,0.25);
    color:white;
}
#result{
    margin-top:15px;
}
input[type="file"]{
    background:white;
    color:black;
}
</style>
</head>

<body>
<div class="container">
<h2>🌽 Corn Disease Classifier</h2>

<video id="video" autoplay playsinline></video>
<img id="preview" style="display:none;">

<div>
<button class="primary" onclick="capture()">📸 ถ่ายภาพ</button>
<button class="secondary" onclick="retake()">🔄 ถ่ายใหม่</button>
<button class="secondary" onclick="classify()">🔍 จำแนก</button>
</div>

<div>
<input type="file" accept="image/*" onchange="uploadImage(event)">
</div>

<div id="result">กำลังโหลดโมเดล...</div>
</div>

<script>
let model;
let video = document.getElementById("video");
let preview = document.getElementById("preview");
let resultText = document.getElementById("result");
let stream;

/* โหลดโมเดล */
async function loadModel(){
    try{
        model = await tmImage.load("model.json","metadata.json");
        resultText.innerHTML = "พร้อมใช้งาน ✅";
        console.log("Model Loaded");
    }catch(err){
        resultText.innerHTML = "โหลดโมเดลไม่สำเร็จ ❌";
        console.error(err);
    }
}

/* เปิดกล้องอัตโนมัติ */
async function startCamera(){
    try{
        stream = await navigator.mediaDevices.getUserMedia({video:true});
        video.srcObject = stream;
    }catch(err){
        alert("กรุณาอนุญาตการใช้กล้อง");
        console.error(err);
    }
}

/* ถ่ายภาพจากกล้อง */
function capture(){
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video,0,0);

    preview.src = canvas.toDataURL();
    preview.style.display="block";
    video.style.display="none";
}

/* ถ่ายใหม่ */
function retake(){
    preview.style.display="none";
    video.style.display="block";
    resultText.innerHTML = "พร้อมใช้งาน ✅";
}

/* อัปโหลดรูป */
function uploadImage(event){
    const file = event.target.files[0];
    if(!file) return;

    const reader = new FileReader();
    reader.onload = function(e){
        preview.src = e.target.result;
        preview.style.display="block";
        video.style.display="none";
    }
    reader.readAsDataURL(file);
}

/* จำแนก */
async function classify(){
    if(!model){
        resultText.innerHTML="โมเดลยังโหลดไม่เสร็จ";
        return;
    }

    if(preview.style.display==="none"){
        resultText.innerHTML="กรุณาถ่ายภาพหรืออัปโหลดรูปก่อน";
        return;
    }

    const prediction = await model.predict(preview);

    let best = prediction.reduce((a,b)=>
        a.probability>b.probability?a:b
    );

    resultText.innerHTML =
        `ผลลัพธ์: <b>${best.className}</b><br>
         ความมั่นใจ: ${(best.probability*100).toFixed(2)}%`;
}

/* เริ่มทำงาน */
window.onload = async () => {
    await loadModel();
    await startCamera();
};
</script>

</body>
</html>
