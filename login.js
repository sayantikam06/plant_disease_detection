let generatedOTP = "";

document.getElementById("otpSection").style.display = "none";

function sendOTP() {
    let phone = document.getElementById("phone").value;

    if (phone.length !== 10) {
        alert("Enter valid 10-digit mobile number");
        return;
    }

    generatedOTP = Math.floor(1000 + Math.random() * 9000);

    alert("Demo OTP: " + generatedOTP);

    document.getElementById("otpSection").style.display = "block";
}

function verifyOTP() {
    let enteredOTP = document.getElementById("otp").value;

    if (enteredOTP == generatedOTP) {
        localStorage.setItem("phone", document.getElementById("phone").value);
        window.location.href = "dashboard.html";
    } else {
        alert("Invalid OTP! Please try again.");
    }
}
