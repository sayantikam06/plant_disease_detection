const phone = localStorage.getItem("phone");

if (!phone) {
    window.location.href = "login.html";
}

document.getElementById("welcome").innerHTML = `Welcome 👋 <br>${phone}`;

const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
    }
});

document.getElementById("detectBtn").addEventListener("click", async() => {
    const file = imageInput.files[0];

    if (!file) {
        alert("Please upload a leaf image first!");
        return;
    }

    document.getElementById("result").innerHTML = `
        <p>🔍 Analyzing image, please wait...</p>
    `;

    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            document.getElementById("result").innerHTML = `
                <p style="color:red;">❌ Error: ${data.error}</p>
            `;
            return;
        }

        const disease = data.disease || data.prediction || "Unknown";
        const confidence = data.confidence ? parseFloat(data.confidence).toFixed(2) : "N/A";

        document.getElementById("result").innerHTML = `
            <h4>🌿 Disease Detected</h4>
            <p><strong>Disease:</strong> ${disease.replace(/_/g, " ")}</p>
            <p><strong>Confidence:</strong> ${confidence}%</p>
        `;

    } catch (error) {
        document.getElementById("result").innerHTML = `
            <p style="color:red;">❌ Could not connect to server. Make sure Flask is running!</p>
        `;
    }
});

function logout() {
    localStorage.clear();
    window.location.href = "login.html";
}