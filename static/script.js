const form = document.getElementById('paraphraseForm');
const submitBtn = document.getElementById('submitBtn');
const btnText = submitBtn.querySelector('.btn-text');
const btnLoader = submitBtn.querySelector('.btn-loader');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const originalTextEl = document.getElementById('originalText');
const paraphrasesList = document.getElementById('paraphrasesList');
const deviceInfo = document.getElementById('deviceInfo');
const errorMessage = document.getElementById('errorMessage');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(form);
    const data = {
        text: formData.get('text'),
        num_return_sequences: parseInt(formData.get('num_return_sequences')),
        num_beams: parseInt(formData.get('num_beams')),
        temperature: parseFloat(formData.get('temperature'))
    };

    setLoading(true);
    hideResults();
    hideError();

    try {
        const response = await fetch('/paraphrase', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'An error occurred');
        }

        displayResults(result);
    } catch (error) {
        displayError(error.message);
    } finally {
        setLoading(false);
    }
});

function setLoading(loading) {
    if (loading) {
        submitBtn.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline';
    } else {
        submitBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

function displayResults(result) {
    originalTextEl.textContent = result.original_text;
    deviceInfo.textContent = result.device;
    
    paraphrasesList.innerHTML = '';
    result.paraphrases.forEach((paraphrase, index) => {
        const li = document.createElement('li');
        li.textContent = paraphrase;
        paraphrasesList.appendChild(li);
    });
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function hideError() {
    errorSection.style.display = 'none';
}
