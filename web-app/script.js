const backendUrl = window.env.BACKEND_URL;

//|| "http://localhost:80"; // Fallback to localhost if not set

console.log("Using backend URL:", backendUrl);

document.getElementById('upload-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const resultDiv = document.getElementById('result');
    resultDiv.textContent = 'Loading...';

    const fileInput = document.getElementById('image');
    const file = fileInput.files[0];
    if (!file) {
        resultDiv.textContent = 'Please upload an image.';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        console.log("Calling back-end with URL:", `${backendUrl}/predict`);

        const response = await fetch(`${backendUrl}/predict`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error("Error response:", errorText);
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();
        resultDiv.textContent = `Predicted Breed: ${data.breed_name}`;
    } catch (error) {
        console.error("Error in fetch:", error);
        resultDiv.textContent = 'Failed to predict the breed. Please try again.';
    }
});