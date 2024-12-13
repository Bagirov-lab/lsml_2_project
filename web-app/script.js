// const backendUrl = window.env.BACKEND_URL;
const backendUrl = "http://core_service:80"; // Hardcoded for testing purposes

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

        console.log("Response status:", response.status);

        ;

        if (!response.ok) {
            const errorText = await response.text();
            console.error("Error response:", errorText);
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();
        console.log("Response data:", data);
        resultDiv.textContent = `Predicted Breed: ${data.breed_name}`;
    } catch (error) {
        console.error("Error in fetch:", error);
        resultDiv.textContent = 'Failed to predict the breed. Please try again.';
    }
});